import os
import logging
import sysconfig

logger = logging.getLogger(__name__)

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from infer.audio import load_audio, load_audio_tensor


_ORT_CUDA_DLL_HANDLES = []


def _configure_ort_cuda_dll_paths():
    """Expose pip-installed CUDA 11/cuDNN 8 DLLs to ONNX Runtime on Windows."""
    if os.name != "nt":
        return

    site_packages = os.path.normpath(sysconfig.get_paths()["purelib"])
    nvidia_root = os.path.join(site_packages, "nvidia")
    dll_dirs = [
        os.path.join(nvidia_root, "cuda_runtime", "bin"),
        os.path.join(nvidia_root, "cublas", "bin"),
        os.path.join(nvidia_root, "cufft", "bin"),
        os.path.join(nvidia_root, "cudnn", "bin"),
        os.path.join(nvidia_root, "cuda_nvrtc", "bin"),
        os.path.join(os.path.dirname(torch.__file__), "lib"),
    ]
    dll_dirs = [path for path in dll_dirs if os.path.isdir(path)]
    if not dll_dirs:
        return

    current_path = os.environ.get("PATH", "")
    current_dirs = [path for path in current_path.split(os.pathsep) if path]
    known_dirs = {os.path.normcase(os.path.normpath(path)) for path in current_dirs}
    prepend_dirs = []
    for path in dll_dirs:
        normalized = os.path.normcase(os.path.normpath(path))
        if normalized not in known_dirs:
            prepend_dirs.append(path)
            known_dirs.add(normalized)
    if prepend_dirs:
        os.environ["PATH"] = os.pathsep.join(prepend_dirs + current_dirs)

    # Python 3.8+ restricts DLL lookup for extension modules. Keep the handles
    # alive for the process lifetime in addition to updating PATH.
    if hasattr(os, "add_dll_directory"):
        for path in dll_dirs:
            try:
                _ORT_CUDA_DLL_HANDLES.append(os.add_dll_directory(path))
            except OSError:
                logger.warning("Unable to add ONNX Runtime DLL directory: %s", path)


_configure_ort_cuda_dll_paths()

cpu = torch.device("cpu")


class ConvTDFNetTrim:
    def __init__(self, device, dim_f, dim_t, n_fft, hop=1024):
        self.dim_f = dim_f
        self.dim_t = 2**dim_t
        self.n_fft = n_fft
        self.hop = hop
        self.n_bins = self.n_fft // 2 + 1
        self.chunk_size = hop * (self.dim_t - 1)
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True).to(device)
        self.dim_c = 4
        self.freq_pad = torch.zeros(
            [1, self.dim_c, self.n_bins - self.dim_f, self.dim_t],
            device=device,
        )

    def stft(self, x):
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            center=True,
            return_complex=True,
        )
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, self.dim_c, self.n_bins, self.dim_t])
        return x[:, :, : self.dim_f]

    def istft(self, x):
        freq_pad = self.freq_pad.expand(x.shape[0], -1, -1, -1)
        x = torch.cat([x, freq_pad], -2)
        c = 2
        x = x.reshape([-1, c, 2, self.n_bins, self.dim_t]).reshape([-1, 2, self.n_bins, self.dim_t])
        x = x.permute([0, 2, 3, 1])
        x = x.contiguous()
        x = torch.view_as_complex(x)
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True)
        return x.reshape([-1, c, self.chunk_size])


def get_models(device, dim_f, dim_t, n_fft):
    return ConvTDFNetTrim(
        device=device,
        dim_f=dim_f,
        dim_t=dim_t,
        n_fft=n_fft,
    )


class Predictor:
    def __init__(self, args):
        import onnxruntime as ort

        available_providers = ort.get_available_providers()
        requested_providers = [
            provider[0] if isinstance(provider, (tuple, list)) else provider
            for provider in args.providers
        ]
        logger.info("ONNX Runtime available providers: %s", available_providers)
        if (
            "CUDAExecutionProvider" in requested_providers
            and "CUDAExecutionProvider" not in available_providers
        ):
            raise RuntimeError(
                "CUDAExecutionProvider is required for the FoxJoy ONNX model, "
                "but the installed ONNX Runtime does not provide it. Install "
                "the matching CUDA ONNX Runtime dependencies with this "
                "project's runtime Python."
            )
        if (
            "DmlExecutionProvider" in requested_providers
            and "DmlExecutionProvider" not in available_providers
        ):
            raise RuntimeError(
                "DmlExecutionProvider is required for the FoxJoy ONNX model, "
                "but the installed ONNX Runtime does not provide it. Install "
                "requirments_cpu_py312.txt with this project's runtime Python."
            )
        self.args = args
        try:
            requested_torch_device = torch.device(args.device)
        except Exception:
            requested_torch_device = cpu
        if requested_torch_device.type == "cuda" and requested_torch_device.index is None:
            requested_torch_device = torch.device("cuda:0")
        # DirectML and CPU keep the established NumPy/CPU STFT path.  The
        # Torch CUDA path is enabled only after the ORT session confirms that
        # its CUDA provider really became the primary provider.
        model_device = requested_torch_device if requested_torch_device.type == "cuda" else cpu
        self.model_ = get_models(
            device=model_device,
            dim_f=args.dim_f,
            dim_t=args.dim_t,
            n_fft=args.n_fft,
        )
        self.model = ort.InferenceSession(
            os.path.join(args.onnx, "vocals.onnx"),
            providers=args.providers,
        )
        active_providers = self.model.get_providers()
        logger.info("ONNX Runtime active providers: %s", active_providers)
        if (
            "CUDAExecutionProvider" in requested_providers
            and (
                not active_providers
                or active_providers[0] != "CUDAExecutionProvider"
            )
        ):
            raise RuntimeError(
                "The FoxJoy ONNX model did not activate CUDAExecutionProvider; "
                "check the CUDA 11/cuDNN 8 DLL installation."
            )
        if (
            "DmlExecutionProvider" in requested_providers
            and (
                not active_providers
                or active_providers[0] != "DmlExecutionProvider"
            )
        ):
            raise RuntimeError(
                "The FoxJoy ONNX model did not activate DmlExecutionProvider; "
                "check the ONNX Runtime DirectML installation."
            )
        self.cuda_pipeline = bool(
            requested_torch_device.type == "cuda"
            and active_providers
            and active_providers[0] == "CUDAExecutionProvider"
        )
        self.torch_device = requested_torch_device if self.cuda_pipeline else cpu
        logger.info(
            "ONNX load done; FoxJoy tensor pipeline=%s, torch device=%s",
            "cuda" if self.cuda_pipeline else "cpu-compatible",
            self.torch_device,
        )

    def _run_ort_cuda(self, input_tensor, output_tensor):
        input_tensor = input_tensor.contiguous()
        if input_tensor.dtype != torch.float32:
            input_tensor = input_tensor.float()
        if not output_tensor.is_contiguous() or output_tensor.dtype != torch.float32:
            raise RuntimeError("FoxJoy CUDA output buffer must be contiguous float32")

        device_id = self.torch_device.index
        io_binding = self.model.io_binding()
        io_binding.bind_input(
            name=self.model.get_inputs()[0].name,
            device_type="cuda",
            device_id=device_id,
            element_type=np.float32,
            shape=tuple(input_tensor.shape),
            buffer_ptr=input_tensor.data_ptr(),
        )
        io_binding.bind_output(
            name=self.model.get_outputs()[0].name,
            device_type="cuda",
            device_id=device_id,
            element_type=np.float32,
            shape=tuple(output_tensor.shape),
            buffer_ptr=output_tensor.data_ptr(),
        )
        # ORT owns a separate CUDA stream by default.  Explicit boundaries
        # guarantee that it sees the completed Torch STFT and that Torch sees
        # the completed output without staging either tensor through NumPy.
        torch.cuda.synchronize(self.torch_device)
        self.model.run_with_iobinding(io_binding)
        torch.cuda.synchronize(self.torch_device)
        return input_tensor

    def _infer_cuda(self, spek):
        spek = spek.contiguous().float()
        output = torch.empty_like(spek)
        if self.args.denoise:
            # Reuse both the ORT output allocation and the input allocation
            # for the negative/positive passes.  Only the accumulator is
            # separate because the second ORT run overwrites its output.
            spek.neg_()
            spek = self._run_ort_cuda(spek, output)
            prediction = output * -0.5
            spek.neg_()
            spek = self._run_ort_cuda(spek, output)
            prediction.add_(output, alpha=0.5)
            return prediction
        self._run_ort_cuda(spek, output)
        return output

    def demix(self, mix):
        samples = mix.shape[-1]
        margin = self.args.margin
        chunk_size = self.args.chunks * 44100
        assert not margin == 0, "margin cannot be zero!"
        if margin > chunk_size:
            margin = chunk_size

        segmented_mix = {}

        if self.args.chunks == 0 or samples < chunk_size:
            chunk_size = samples

        counter = -1
        for skip in range(0, samples, chunk_size):
            counter += 1

            s_margin = 0 if counter == 0 else margin
            end = min(skip + chunk_size + margin, samples)

            start = skip - s_margin

            segment = mix[:, start:end]
            # CUDA segments are views of the already resident decoded audio;
            # copying every segment would almost double long-file VRAM use.
            segmented_mix[skip] = segment if torch.is_tensor(segment) else segment.copy()
            if end == samples:
                break

        sources = self.demix_base(segmented_mix, margin_size=margin)
        """
        mix:(2,big_sample)
        segmented_mix:offset->(2,small_sample)
        sources:(1,2,big_sample)
        """
        return sources

    def demix_base(self, mixes, margin_size):
        chunked_sources = []
        progress_bar = tqdm(total=len(mixes))
        progress_bar.set_description("Processing")
        for mix in mixes:
            cmix = mixes[mix]
            sources = []
            n_sample = cmix.shape[1]
            model = self.model_
            trim = model.n_fft // 2
            gen_size = model.chunk_size - 2 * trim
            pad = gen_size - n_sample % gen_size
            if self.cuda_pipeline and torch.is_tensor(cmix):
                cmix = cmix.to(self.torch_device, dtype=torch.float32)
                mix_p = torch.cat(
                    (
                        cmix.new_zeros((2, trim)),
                        cmix,
                        cmix.new_zeros((2, pad)),
                        cmix.new_zeros((2, trim)),
                    ),
                    1,
                )
            else:
                mix_p = np.concatenate(
                    (
                        np.zeros((2, trim)),
                        cmix,
                        np.zeros((2, pad)),
                        np.zeros((2, trim)),
                    ),
                    1,
                )
            mix_waves = []
            i = 0
            while i < n_sample + pad:
                waves = mix_p[:, i : i + model.chunk_size]
                if not torch.is_tensor(waves):
                    waves = np.array(waves)
                mix_waves.append(waves)
                i += gen_size
            if torch.is_tensor(mix_waves[0]):
                mix_waves = torch.stack(mix_waves).float()
            else:
                mix_waves = torch.from_numpy(np.asarray(mix_waves, dtype=np.float32))
            with torch.no_grad():
                _ort = self.model
                if self.cuda_pipeline:
                    # One H2D for all windows in this outer segment. STFT,
                    # both denoise passes and ISTFT remain on the selected
                    # CUDA device; only the finished waveform returns to CPU.
                    if mix_waves.device != self.torch_device:
                        mix_waves = mix_waves.to(self.torch_device, non_blocking=True)
                    spek = model.stft(mix_waves)
                    spec_pred = self._infer_cuda(spek)
                    tar_waves = model.istft(spec_pred)
                    tar_signal = (
                        tar_waves[:, :, trim:-trim]
                        .transpose(0, 1)
                        .reshape(2, -1)[:, :-pad]
                        .cpu()
                        .numpy()
                    )
                else:
                    spek = model.stft(mix_waves)
                    if self.args.denoise:
                        spek_numpy = spek.numpy()
                        spec_pred = (
                            -_ort.run(None, {"input": -spek_numpy})[0] * 0.5
                            + _ort.run(None, {"input": spek_numpy})[0] * 0.5
                        )
                        tar_waves = model.istft(torch.from_numpy(spec_pred))
                    else:
                        spec_pred = _ort.run(None, {"input": spek.numpy()})[0]
                        tar_waves = model.istft(torch.from_numpy(spec_pred))
                    tar_signal = (
                        tar_waves[:, :, trim:-trim]
                        .transpose(0, 1)
                        .reshape(2, -1)
                        .numpy()[:, :-pad]
                    )

                start = 0 if mix == 0 else margin_size
                end = None if mix == list(mixes.keys())[::-1][0] else -margin_size
                sources.append(tar_signal[:, start:end])

                progress_bar.update(1)

            chunked_sources.append(sources)
        _sources = np.concatenate(chunked_sources, axis=-1)
        # del self.model
        progress_bar.close()
        return _sources

    def prediction(self, m, vocal_root, others_root, format):
        os.makedirs(vocal_root, exist_ok=True)
        os.makedirs(others_root, exist_ok=True)
        basename = os.path.basename(m)
        if self.cuda_pipeline:
            mix = load_audio_tensor(m, 44100, force_mono=False)
            mix = mix.to(self.torch_device)
        else:
            mix = load_audio(m, 44100, force_mono=False)
        rate = 44100
        if mix.ndim == 1:
            mix = mix.unsqueeze(0) if torch.is_tensor(mix) else mix[np.newaxis, :]
        if mix.shape[0] == 1:
            mix = mix.repeat(2, 1) if torch.is_tensor(mix) else np.repeat(mix, 2, axis=0)
        elif mix.shape[0] > 2:
            mix = mix[:2].contiguous() if torch.is_tensor(mix) else np.ascontiguousarray(mix[:2])
        sources = self.demix(mix)
        opt = sources[0].T
        if torch.is_tensor(mix):
            mix = mix.transpose(0, 1).float().cpu().numpy()
        else:
            mix = mix.T
        if format in ["wav", "flac"]:
            sf.write("%s/%s_main_vocal.%s" % (vocal_root, basename, format), mix - opt, rate)
            sf.write("%s/%s_others.%s" % (others_root, basename, format), opt, rate)
        else:
            path_vocal = "%s/%s_main_vocal.wav" % (vocal_root, basename)
            path_other = "%s/%s_others.wav" % (others_root, basename)
            sf.write(path_vocal, mix - opt, rate)
            sf.write(path_other, opt, rate)
            opt_path_vocal = path_vocal[:-4] + ".%s" % format
            opt_path_other = path_other[:-4] + ".%s" % format
            if os.path.exists(path_vocal):
                os.system('ffmpeg -i "%s" -vn "%s" -q:a 2 -y' % (path_vocal, opt_path_vocal))
                if os.path.exists(opt_path_vocal):
                    try:
                        os.remove(path_vocal)
                    except:
                        pass
            if os.path.exists(path_other):
                os.system('ffmpeg -i "%s" -vn "%s" -q:a 2 -y' % (path_other, opt_path_other))
                if os.path.exists(opt_path_other):
                    try:
                        os.remove(path_other)
                    except:
                        pass


class MDXNetDereverb:
    def __init__(self, chunks, providers, device="cpu"):
        self.onnx = os.path.join(
            os.getenv("weight_uvr5_root", "assets/uvr5_weights"),
            "onnx_dereverb_By_FoxJoy",
        )
        self.chunks = chunks
        self.providers = providers
        self.device = device
        self.margin = 44100
        self.dim_t = 9
        self.dim_f = 3072
        self.n_fft = 6144
        self.denoise = True
        self.pred = Predictor(self)

    def _path_audio_(self, input, others_root, vocal_root, format, is_hp3=False):
        self.pred.prediction(input, vocal_root, others_root, format)
