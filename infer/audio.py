import platform, os
import ffmpeg
import numpy as np
import av
from io import BytesIO
import threading


_USE_TORCHAUDIO_GPU = False
_AUDIO_DEVICE = None
_AUDIO_DTYPE = None
_TORCH = None
_TORCHAUDIO = None
_TORCHAUDIO_RESAMPLE = None
_RESAMPLE_TRANSFORMS = {}
_RESAMPLE_LOCK = threading.Lock()
_FORCE_CPU_AUDIO = os.environ.get("RVC_AUDIO_FORCE_CPU", "0") == "1"

# Select the accelerated loader once, when this module is imported.  The CUDA
# device and dtype come from the project's shared automatic selection rules.
# CPU and DirectML keep the original FFmpeg path.  Import failures (including
# missing torchaudio DLLs) also leave FFmpeg selected.
if not _FORCE_CPU_AUDIO:
    try:
        import torch as _TORCH
        import torchaudio as _TORCHAUDIO
        from torchaudio.transforms import Resample as _TORCHAUDIO_RESAMPLE

        from configs.config import infer_device as _AUDIO_DEVICE
        from configs.config import infer_dtype as _AUDIO_DTYPE

        _USE_TORCHAUDIO_GPU = (
            getattr(_AUDIO_DEVICE, "type", None) == "cuda"
            and _TORCH.cuda.is_available()
        )
    except Exception:
        _USE_TORCHAUDIO_GPU = False

AUDIO_LOAD_BACKEND = "torchaudio_cuda" if _USE_TORCHAUDIO_GPU else "ffmpeg"
TORCHAUDIO_GPU_ENABLED = _USE_TORCHAUDIO_GPU
AUDIO_DEVICE = _AUDIO_DEVICE
AUDIO_DTYPE = _AUDIO_DTYPE


def wav2(i, o, format):
    inp = av.open(i, "r")
    try:
        if format == "m4a":
            format = "mp4"
        out = av.open(o, "w", format=format)
        try:
            if format == "ogg":
                format = "libvorbis"
            if format == "mp4":
                format = "aac"

            if not inp.streams.audio:
                raise ValueError("Input contains no audio stream")
            input_stream = inp.streams.audio[0]
            source_rate = input_stream.codec_context.sample_rate
            ostream = (
                out.add_stream(format, rate=source_rate)
                if source_rate
                else out.add_stream(format)
            )
            source_channels = input_stream.codec_context.channels
            if source_channels == 1:
                ostream.layout = "mono"
            elif source_channels == 2:
                ostream.layout = "stereo"

            for frame in inp.decode(input_stream):
                for p in ostream.encode(frame):
                    out.mux(p)

            for p in ostream.encode(None):
                out.mux(p)
        finally:
            out.close()
    finally:
        inp.close()


def transcode_audio_file(input_path, output_path, format):
    """Transcode a WAV path and remove partial compressed output on failure."""
    output_path = os.fspath(output_path)
    if os.path.exists(output_path):
        os.remove(output_path)
    try:
        wav2(input_path, output_path, format)
        if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("Audio transcoding produced no output: %s" % output_path)
    except Exception:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise


def _probe_audio(file):
    info = ffmpeg.probe(file, cmd="ffprobe")
    stream = next(
        item for item in info.get("streams", []) if item.get("codec_type") == "audio"
    )
    return int(stream["sample_rate"]), int(stream.get("channels", 1))


def _decode_audio_ffmpeg(file):
    """Decode one audio stream without changing its sample rate."""
    source_sr, channels = _probe_audio(file)
    channels = max(1, channels)
    out, _ = (
        ffmpeg.input(file, threads=0)
        .output(
            "-",
            format="f32le",
            acodec="pcm_f32le",
            ac=channels,
            ar=source_sr,
        )
        .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
    )
    samples = np.frombuffer(out, np.float32)
    usable = samples.size - samples.size % channels
    samples = samples[:usable]
    return samples.reshape(-1, channels).T.copy(), source_sr


def _load_audio_ffmpeg(file, sr, force_mono=True):
    # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
    # Keep the original FFmpeg path for CPU and unsupported decoders.  When
    # stereo is requested, retain all source channels instead of forcing ac=1.
    channels = 1
    if not force_mono:
        _, channels = _probe_audio(file)
        channels = max(1, channels)
    output = {
        "format": "f32le",
        "acodec": "pcm_f32le",
        "ar": sr,
    }
    if force_mono:
        output["ac"] = 1
    else:
        output["ac"] = channels
    out, _ = (
        ffmpeg.input(file, threads=0)
        .output("-", **output)
        .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
    )
    samples = np.frombuffer(out, np.float32)
    if force_mono:
        return samples.flatten()
    usable = samples.size - samples.size % channels
    return samples[:usable].reshape(-1, channels).T.copy()


def _channel_first_tensor(audio):
    if not _TORCH.is_tensor(audio):
        audio = _TORCH.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.ndim != 2:
        raise ValueError("Audio data must be one-dimensional or channel-first two-dimensional")
    return audio


def _format_audio_tensor(audio, force_mono=True, keep_on_device=False):
    audio = _channel_first_tensor(audio)
    if force_mono:
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        else:
            audio = audio[:1]
    if keep_on_device:
        audio = audio.to(device=_AUDIO_DEVICE, dtype=_AUDIO_DTYPE)
        return audio.flatten() if force_mono else audio
    audio = audio.detach().to(device="cpu", dtype=_TORCH.float32).contiguous().numpy()
    return audio[0].flatten() if force_mono else audio


def _get_gpu_resampler(source_sr, target_sr):
    key = (source_sr, target_sr, str(_AUDIO_DEVICE), _AUDIO_DTYPE)
    transform = _RESAMPLE_TRANSFORMS.get(key)
    if transform is None:
        with _RESAMPLE_LOCK:
            transform = _RESAMPLE_TRANSFORMS.get(key)
            if transform is None:
                transform = _TORCHAUDIO_RESAMPLE(
                    source_sr,
                    target_sr,
                    dtype=_AUDIO_DTYPE,
                ).to(device=_AUDIO_DEVICE, dtype=_AUDIO_DTYPE)
                _RESAMPLE_TRANSFORMS[key] = transform
    return transform


def _resample_tensor_gpu(audio, source_sr, target_sr, force_mono=True, keep_on_device=False):
    audio = _channel_first_tensor(audio)
    if source_sr == target_sr:
        return _format_audio_tensor(audio, force_mono, keep_on_device)
    with _TORCH.inference_mode():
        audio = audio.to(device=_AUDIO_DEVICE, dtype=_AUDIO_DTYPE)
        if force_mono and audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        elif force_mono:
            audio = audio[:1]
        audio = _get_gpu_resampler(source_sr, target_sr)(audio)
        if keep_on_device:
            return audio.flatten() if force_mono else audio
        audio = audio.detach().to(device="cpu", dtype=_TORCH.float32).contiguous().numpy()
        return audio[0].flatten() if force_mono else audio


def _load_audio_torchaudio_gpu(file, sr, force_mono=True, keep_on_device=False):
    audio, source_sr = _TORCHAUDIO.load(file)
    return _resample_tensor_gpu(audio, source_sr, sr, force_mono, keep_on_device)


def resample_audio(audio, source_sr, target_sr, force_mono=False, res_type=None):
    """Resample channel-first audio, using the same CUDA path as load_audio."""
    tensor = _channel_first_tensor(audio)
    if _USE_TORCHAUDIO_GPU:
        try:
            return _resample_tensor_gpu(
                tensor, source_sr, target_sr, force_mono, keep_on_device=False
            )
        except Exception:
            try:
                _TORCH.cuda.empty_cache()
            except Exception:
                pass

    tensor = _channel_first_tensor(tensor)
    if force_mono:
        if tensor.shape[0] > 1:
            tensor = tensor.mean(dim=0, keepdim=True)
        else:
            tensor = tensor[:1]
    array = tensor.detach().to(device="cpu", dtype=_TORCH.float32).numpy()
    if source_sr != target_sr:
        import librosa

        kwargs = {}
        if res_type is not None:
            kwargs["res_type"] = res_type
        array = librosa.resample(
            array,
            orig_sr=source_sr,
            target_sr=target_sr,
            axis=-1,
            **kwargs,
        )
    return array[0].flatten() if force_mono else array


def resample_audio_tensor(audio, source_sr, target_sr, force_mono=False):
    """Resample channel-first audio and keep the result on the selected GPU."""
    if _USE_TORCHAUDIO_GPU:
        try:
            return _resample_tensor_gpu(
                audio,
                source_sr,
                target_sr,
                force_mono=force_mono,
                keep_on_device=True,
            )
        except Exception:
            try:
                _TORCH.cuda.empty_cache()
            except Exception:
                pass
    result = resample_audio(
        audio,
        source_sr,
        target_sr,
        force_mono=force_mono,
    )
    tensor = _TORCH.from_numpy(np.ascontiguousarray(result, dtype=np.float32))
    if _USE_TORCHAUDIO_GPU:
        tensor = tensor.to(device=_AUDIO_DEVICE, dtype=_AUDIO_DTYPE)
    return tensor


def load_audio_tensor(file, sr, force_mono=True):
    """Load audio as a tensor; stereo mode returns channel-first data."""
    file = clean_path(file)
    if not _USE_TORCHAUDIO_GPU:
        return _TORCH.from_numpy(_load_audio_ffmpeg(file, sr, force_mono))
    try:
        return _load_audio_torchaudio_gpu(
            file, sr, force_mono=force_mono, keep_on_device=True
        )
    except Exception as torchaudio_error:
        try:
            audio, source_sr = _decode_audio_ffmpeg(file)
            return _resample_tensor_gpu(
                audio, source_sr, sr, force_mono=force_mono, keep_on_device=True
            )
        except Exception as ffmpeg_error:
            try:
                audio = _TORCH.from_numpy(_load_audio_ffmpeg(file, sr, force_mono))
                return audio.to(device=_AUDIO_DEVICE, dtype=_AUDIO_DTYPE)
            except Exception:
                raise RuntimeError(
                    "Failed to load audio with torchaudio (%s) and FFmpeg (%s)"
                    % (torchaudio_error, ffmpeg_error)
                ) from ffmpeg_error


def load_audio(file, sr, force_mono=True):
    """Load float32 audio; mono is [T], preserved channels are [C, T]."""
    file = clean_path(file)  # 防止小白拷路径头尾带了空格和"和回车
    if _USE_TORCHAUDIO_GPU:
        try:
            return _load_audio_torchaudio_gpu(file, sr, force_mono=force_mono)
        except Exception as torchaudio_error:
            try:
                audio, source_sr = _decode_audio_ffmpeg(file)
                return _resample_tensor_gpu(
                    audio, source_sr, sr, force_mono=force_mono
                )
            except Exception as decode_error:
                # Preserve the old format coverage if both torchaudio decode
                # and GPU processing are unavailable for this file.
                try:
                    return _load_audio_ffmpeg(file, sr, force_mono)
                except Exception as ffmpeg_error:
                    raise RuntimeError(
                        "Failed to load audio with torchaudio (%s), raw FFmpeg (%s), and FFmpeg (%s)"
                        % (torchaudio_error, decode_error, ffmpeg_error)
                    ) from ffmpeg_error
    try:
        return _load_audio_ffmpeg(file, sr, force_mono)
    except Exception as error:
        raise RuntimeError("Failed to load audio: %s" % error) from error


def clean_path(path_str):
    if platform.system() == "Windows":
        path_str = path_str.replace("/", "\\")
    return path_str.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
