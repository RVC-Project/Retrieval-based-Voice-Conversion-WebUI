import json
import subprocess

import av
import numpy as np


def _frame_to_audio(frame, mono):
    """Implement the frame to audio helper.

    Args:
        frame (Any): Frame value.
        mono (bool): Mono value.

    Returns:
        Any: Computed result."""
    audio = frame.to_ndarray()
    audio = audio[None, :] if audio.ndim == 1 else audio
    return (audio.mean(axis=0, keepdims=True) if mono and audio.shape[0] > 1 else audio).astype(np.float32, copy=False)


def _ffmpeg_audio_stream_info(path):
    """Return basic audio stream information from ffprobe."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout or "{}").get("streams") or []
    if not streams:
        raise ValueError(f"No audio stream found in {path!s}.")
    stream = streams[0]
    return int(stream["sample_rate"]), int(stream["channels"])


def _load_audio_ffmpeg(path, sr=None, mono=False, offset=0.0, duration=None):
    """Load audio through the ffmpeg CLI as a fallback for damaged streams."""
    source_rate, source_channels = _ffmpeg_audio_stream_info(path)
    out_rate = int(sr or source_rate)
    channels = 1 if mono else source_channels
    command = ["ffmpeg", "-nostdin", "-v", "error"]
    if offset:
        command += ["-ss", str(float(offset))]
    command += ["-i", str(path), "-map", "0:a:0", "-vn"]
    if duration is not None:
        command += ["-t", str(float(duration))]
    command += ["-f", "f32le", "-acodec", "pcm_f32le", "-ar", str(out_rate), "-ac", str(channels), "-"]
    result = subprocess.run(command, check=True, capture_output=True)
    audio = np.frombuffer(result.stdout, dtype="<f4")
    complete_samples = audio.size // channels
    audio = audio[: complete_samples * channels]
    audio = audio.reshape(complete_samples, channels).T
    audio = np.ascontiguousarray(audio.astype(np.float32, copy=False))
    return (audio[0] if mono or channels == 1 else audio), out_rate


def _load_audio_librosa(path, sr=None, mono=False, offset=0.0, duration=None):
    """Load audio through librosa as a fallback before the ffmpeg CLI."""
    import librosa

    audio, out_rate = librosa.load(
        path,
        sr=sr,
        mono=mono,
        offset=float(offset or 0.0),
        duration=None if duration is None else float(duration),
        dtype=np.float32,
    )
    audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32))
    return (audio[0] if audio.ndim > 1 and (mono or audio.shape[0] == 1) else audio), int(out_rate)


def _load_audio_av(path, sr=None, mono=False, offset=0.0, duration=None):
    """Load audio through PyAV."""
    chunks = []
    out_rate = None
    with av.open(path) as container:
        stream = container.streams.audio[0]
        out_rate = int(sr or stream.rate)
        resampler = None
        stop_samples = None if duration is None else int(round((offset + duration) * out_rate))
        decoded = 0

        for frame in container.decode(stream):
            if resampler is None:
                resampler = av.AudioResampler(format="fltp", layout=frame.layout.name, rate=out_rate)

            for out in resampler.resample(frame):
                chunks.append(audio := _frame_to_audio(out, mono))
                decoded += audio.shape[-1]
            if stop_samples is not None and decoded >= stop_samples:
                break

        if resampler is not None:
            for out in resampler.resample(None):
                chunks.append(_frame_to_audio(out, mono))

    start = int(round(offset * out_rate))
    stop = None if duration is None else start + int(round(duration * out_rate))
    channels = 1 if mono else 0
    audio = np.ascontiguousarray(
        (np.concatenate(chunks, axis=-1) if chunks else np.empty((channels, 0), dtype=np.float32))[..., start:stop]
    )
    return (audio[0] if mono or audio.shape[0] == 1 else audio), out_rate


def load_audio(path, sr=None, mono=False, offset=0.0, duration=None):
    """Load an audio file as float32 NumPy samples.

    Audio decoding is attempted in this order: PyAV, librosa, then the
    ffmpeg CLI fallback. Stereo or multi-channel output is returned
    channel-first as ``(channels, samples)``. Mono output is returned as a
    one-dimensional array.

    Args:
        path (str | os.PathLike): Input audio file path. Any format supported
            by the local FFmpeg/PyAV build can be decoded.
        sr (int | None, optional): Target sample rate. ``None`` keeps the
            source stream sample rate. Defaults to None.
        mono (bool, optional): Whether to downmix multi-channel audio to mono.
            Defaults to False.
        offset (float, optional): Start offset in seconds. Defaults to 0.0.
        duration (float | None, optional): Maximum duration to return in
            seconds after ``offset``. ``None`` reads to the end. Defaults to
            None.

    Returns:
        tuple[np.ndarray, int]: Audio samples and sample rate. The array is
        channel-first for multi-channel audio and one-dimensional for mono
        output.

    Example:
        >>> from pymss import load_audio
        >>> audio, sample_rate = load_audio("song.wav", sr=44100)
        >>> sample_rate
        44100

    Example:
        >>> clip, sample_rate = load_audio(
        ...     "song.wav",
        ...     mono=True,
        ...     offset=30.0,
        ...     duration=10.0,
        ... )
        >>> clip.ndim
        1"""
    loaders = [
        ("PyAV", _load_audio_av),
        ("librosa", _load_audio_librosa),
        ("ffmpeg CLI", _load_audio_ffmpeg),
    ]

    errors = []
    for name, loader in loaders:
        try:
            return loader(path, sr=sr, mono=mono, offset=offset, duration=duration)
        except Exception as e:
            errors.append(f"{name}: {e}")
            continue

    raise RuntimeError(f"All audio loading methods failed: {'; '.join(errors)}")


def _bitrate_to_int(value):
    """Implement the bitrate to int helper.

    Args:
        value (Any): Value value.

    Returns:
        Any: Computed result."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    value = str(value).strip().lower()
    return int(float(value[:-1]) * 1000) if value.endswith("k") else int(value)


def _format_audio(audio):
    """Format audio.

    Args:
        audio (np.ndarray): Audio samples.

    Returns:
        Any: Computed result."""
    audio = np.asarray(audio)
    audio = np.ascontiguousarray(audio[:, None] if audio.ndim == 1 else audio)
    # We can use "fltp" container for all output formats, while the final result is determined by the codec.
    # Using the fltp sample format can also help avoid some clipping distortion that occurs with integer formats.
    return np.ascontiguousarray(audio.astype(np.float32).T)


def save_audio(path, audio, sr, output_format, audio_params):
    """Save a NumPy audio array to wav, flac, mp3, or m4a.

    Audio is expected as sample-major data, either ``(samples,)`` for mono or
    ``(samples, channels)`` for multi-channel audio. The output codec is chosen
    from ``output_format`` and ``audio_params``.

    Args:
        path (str | os.PathLike): Output file path.
        audio (np.ndarray): Audio samples. Mono arrays may be one-dimensional;
            stereo arrays should be shaped as ``(samples, 2)``.
        sr (int): Sample rate in Hz.
        output_format (str): Output format. Supported values are ``wav``,
            ``flac``, ``mp3``, and ``m4a``.
        audio_params (dict): Encoding options. Supported keys include
            ``wav_bit_depth`` (``FLOAT``, ``PCM_16``, ``PCM_24``),
            ``flac_bit_depth`` (currently ``PCM_24`` uses soundfile),
            ``mp3_bit_rate`` (for example ``"320k"``), ``m4a_bit_rate``,
            ``m4a_codec``, and ``m4a_aac_at_quality``.

    Returns:
        None: The file is written to ``path``.

    Example:
        >>> from pymss import save_audio
        >>> save_audio(
        ...     "vocals.wav",
        ...     vocals,
        ...     44100,
        ...     "wav",
        ...     {"wav_bit_depth": "FLOAT"},
        ... )

    Example:
        >>> save_audio(
        ...     "instrumental.flac",
        ...     instrumental,
        ...     44100,
        ...     "flac",
        ...     {"flac_bit_depth": "PCM_24"},
        ... )"""
    output_format = output_format.lower()
    audio_array = np.asarray(audio)
    layout = "stereo" if audio_array.ndim > 1 and audio_array.shape[1] == 2 else "mono"

    if output_format == "mp3":
        codec = "libmp3lame"
    elif output_format == "m4a":
        codec = audio_params.get("m4a_codec", "aac")
    elif output_format == "flac":
        # PyAV's FLAC encoder only exposes a single "flac" codec in the current version.
        # In the current version, without access to bits_per_raw_sample in PyAV, PCM_24 may still be encoded as 16-bit.
        # Use soundfile to save 24-bit FLAC
        codec = "flac"
        if audio_params.get("flac_bit_depth", "PCM_24") == "PCM_24":
            import soundfile as sf

            return sf.write(path, audio_array, int(sr), format="FLAC", subtype="PCM_24")
    else:
        wav_codecs = {"PCM_16": "pcm_s16le", "PCM_24": "pcm_s24le", "FLOAT": "pcm_f32le"}
        codec = wav_codecs.get(audio_params.get("wav_bit_depth", "FLOAT"), wav_codecs["FLOAT"])

    with av.open(path, "w") as container:
        stream = container.add_stream(codec, rate=int(sr))
        stream.layout = layout
        if output_format == "mp3":
            stream.bit_rate = _bitrate_to_int(audio_params.get("mp3_bit_rate", "320k"))
        elif output_format == "m4a":
            stream.bit_rate = _bitrate_to_int(audio_params.get("m4a_bit_rate", "512k"))
            if codec == "aac_at":
                stream.codec_context.options = {"aac_at_quality": str(audio_params.get("m4a_aac_at_quality", 2))}

        frame = av.AudioFrame.from_ndarray(_format_audio(audio_array), format="fltp", layout=layout)
        frame.sample_rate = int(sr)
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
