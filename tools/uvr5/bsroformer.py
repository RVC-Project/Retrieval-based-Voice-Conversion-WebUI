# This code is modified from https://github.com/ZFTurbo/
import os
import warnings
from contextlib import nullcontext

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import yaml

from infer.audio import TORCHAUDIO_GPU_ENABLED, load_audio, load_audio_tensor
from tqdm import tqdm
from tools.file_io import read_text
from i18n.i18n import I18nAuto

warnings.filterwarnings("ignore")
i18n = I18nAuto()


class Roformer_Loader:
    def get_config(self, config_path):
        return yaml.load(read_text(config_path), Loader=yaml.FullLoader)

    def get_default_config(self):
        default_config = None
        if self.model_type == "bs_roformer":
            # Use model_bs_roformer_ep_368_sdr_12.9628.yaml and model_bs_roformer_ep_317_sdr_12.9755.yaml as default configuration files
            # Other BS_Roformer models may not be compatible
            # fmt: off
            default_config = {
                "audio": {"chunk_size": 352800, "sample_rate": 44100},
                "model": {
                    "dim": 512,
                    "depth": 12,
                    "stereo": True,
                    "num_stems": 1,
                    "time_transformer_depth": 1,
                    "freq_transformer_depth": 1,
                    "linear_transformer_depth": 0,
                    "freqs_per_bands": (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12, 24, 24, 24, 24, 24, 24, 24, 24, 48, 48, 48, 48, 48, 48, 48, 48, 128, 129),
                    "dim_head": 64,
                    "heads": 8,
                    "attn_dropout": 0.1,
                    "ff_dropout": 0.1,
                    "flash_attn": True,
                    "dim_freqs_in": 1025,
                    "stft_n_fft": 2048,
                    "stft_hop_length": 441,
                    "stft_win_length": 2048,
                    "stft_normalized": False,
                    "mask_estimator_depth": 2,
                    "multi_stft_resolution_loss_weight": 1.0,
                    "multi_stft_resolutions_window_sizes": (4096, 2048, 1024, 512, 256),
                    "multi_stft_hop_size": 147,
                    "multi_stft_normalized": False,
                },
                "training": {"instruments": ["vocals", "other"], "target_instrument": "vocals"},
                "inference": {"batch_size": 2, "num_overlap": 2},
            }
            # fmt: on
        elif self.model_type == "mel_band_roformer":
            # Use model_mel_band_roformer_ep_3005_sdr_11.4360.yaml as default configuration files
            # Other Mel_Band_Roformer models may not be compatible
            default_config = {
                "audio": {"chunk_size": 352800, "sample_rate": 44100},
                "model": {
                    "dim": 384,
                    "depth": 12,
                    "stereo": True,
                    "num_stems": 1,
                    "time_transformer_depth": 1,
                    "freq_transformer_depth": 1,
                    "linear_transformer_depth": 0,
                    "num_bands": 60,
                    "dim_head": 64,
                    "heads": 8,
                    "attn_dropout": 0.1,
                    "ff_dropout": 0.1,
                    "flash_attn": True,
                    "dim_freqs_in": 1025,
                    "sample_rate": 44100,
                    "stft_n_fft": 2048,
                    "stft_hop_length": 441,
                    "stft_win_length": 2048,
                    "stft_normalized": False,
                    "mask_estimator_depth": 2,
                    "multi_stft_resolution_loss_weight": 1.0,
                    "multi_stft_resolutions_window_sizes": (4096, 2048, 1024, 512, 256),
                    "multi_stft_hop_size": 147,
                    "multi_stft_normalized": False,
                },
                "training": {"instruments": ["vocals", "other"], "target_instrument": "vocals"},
                "inference": {"batch_size": 2, "num_overlap": 2},
            }

        return default_config

    def get_model_from_config(self):
        if self.model_type == "bs_roformer":
            from tools.uvr5.bs_roformer.bs_roformer import BSRoformer

            model = BSRoformer(**dict(self.config["model"]))
        elif self.model_type == "mel_band_roformer":
            from tools.uvr5.bs_roformer.mel_band_roformer import MelBandRoformer

            model = MelBandRoformer(**dict(self.config["model"]))
        else:
            print(i18n("错误：未知模型：%s") % self.model_type)
            model = None
        return model

    def demix_track(self, model, mix, device):
        C = self.config["audio"]["chunk_size"]  # chunk_size
        N = self.config["inference"]["num_overlap"]
        fade_size = C // 10
        step = int(C // N)
        border = C - step
        batch_size = self.config["inference"]["batch_size"]

        length_init = mix.shape[-1]

        # Do pad from the beginning and end to account floating window results better
        if length_init > 2 * border and (border > 0):
            mix = nn.functional.pad(mix, (border, border), mode="reflect")
        total_windows = (mix.shape[-1] + step - 1) // step
        progress_bar = tqdm(total=total_windows, desc="Processing", leave=False)

        parsed_device = device if isinstance(device, torch.device) else torch.device(device)
        device_type = parsed_device.type
        if self.config["training"]["target_instrument"] is None:
            source_count = len(self.config["training"]["instruments"])
        else:
            source_count = 1
        req_shape = (source_count,) + tuple(mix.shape)

        accumulation_device = torch.device("cpu")
        if device_type == "cuda":
            required_bytes = int(np.prod(req_shape)) * 4 + mix.shape[-1] * 4
            free_bytes, _ = torch.cuda.mem_get_info(parsed_device)
            limit = min(1024**3, int(free_bytes * 0.22))
            if required_bytes <= limit:
                accumulation_device = parsed_device

        try:
            result = torch.zeros(
                req_shape,
                dtype=torch.float32,
                device=accumulation_device,
            )
            counter = torch.zeros(
                mix.shape[-1],
                dtype=torch.float32,
                device=accumulation_device,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            accumulation_device = torch.device("cpu")
            result = torch.zeros(req_shape, dtype=torch.float32)
            counter = torch.zeros(mix.shape[-1], dtype=torch.float32)

        # The overlap-add window lives beside the accumulator.  A short file
        # with one window uses the all-ones window to avoid a zero denominator.
        fadein = torch.linspace(
            0, 1, fade_size, device=accumulation_device, dtype=torch.float32
        )
        fadeout = torch.linspace(
            1, 0, fade_size, device=accumulation_device, dtype=torch.float32
        )
        window_full = torch.ones(C, device=accumulation_device)
        window_start = window_full.clone()
        window_middle = window_full.clone()
        window_finish = window_full.clone()
        window_start[-fade_size:] *= fadeout
        window_finish[:fade_size] *= fadein
        window_middle[-fade_size:] *= fadeout
        window_middle[:fade_size] *= fadein

        amp_context = (
            torch.amp.autocast("cuda", enabled=self.is_half)
            if device_type == "cuda"
            else nullcontext()
        )
        grad_context = (
            torch.no_grad()
            if device_type == "privateuseone"
            else torch.inference_mode()
        )
        with amp_context:
            # DirectML updates version counters in several linear kernels and
            # therefore needs no_grad rather than inference_mode. CUDA and CPU
            # retain the existing inference-mode path.
            with grad_context:
                model_dtype = next(model.parameters()).dtype
                i = 0
                batch_data = []
                batch_locations = []
                while i < mix.shape[1]:
                    part = mix[:, i : i + C]
                    length = part.shape[-1]
                    if length < C:
                        if length > C // 2 + 1:
                            part = nn.functional.pad(input=part, pad=(0, C - length), mode="reflect")
                        else:
                            part = nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode="constant", value=0)
                    batch_data.append(part)
                    batch_locations.append((i, length))
                    i += step
                    progress_bar.update(1)

                    if len(batch_data) >= batch_size or (i >= mix.shape[1]):
                        arr = torch.stack(batch_data, dim=0).to(
                            device=parsed_device,
                            dtype=model_dtype,
                        )
                        # Torch STFT/ISTFT cannot be captured reliably by a
                        # CUDA Graph on the supported runtime, so keep this
                        # model call eager while all tensors remain on CUDA.
                        x = model(arr)
                        x_for_accumulation = (
                            x.float()
                            if accumulation_device.type == "cuda"
                            else x.float().cpu()
                        )
                        for j in range(len(batch_locations)):
                            start, l = batch_locations[j]
                            is_first = start == 0
                            is_last = start + l >= mix.shape[1]
                            if is_first and is_last:
                                window = window_full
                            elif is_first:
                                window = window_start
                            elif is_last:
                                window = window_finish
                            else:
                                window = window_middle
                            result[..., start : start + l].add_(
                                x_for_accumulation[j][..., :l] * window[:l]
                            )
                            counter[start : start + l].add_(window[:l])

                        batch_data = []
                        batch_locations = []

                result.div_(counter.clamp_min(1e-8))
                torch.nan_to_num_(result)
                if length_init > 2 * border and (border > 0):
                    result = result[..., border:-border]
                estimated_sources = result.cpu().numpy()

        progress_bar.close()

        if self.config["training"]["target_instrument"] is None:
            return {k: v for k, v in zip(self.config["training"]["instruments"], estimated_sources)}
        else:
            return {k: v for k, v in zip([self.config["training"]["target_instrument"]], estimated_sources)}

    def run_folder(self, input, vocal_root, others_root, format):
        self.model.eval()
        path = input
        os.makedirs(vocal_root, exist_ok=True)
        os.makedirs(others_root, exist_ok=True)
        file_base_name = os.path.splitext(os.path.basename(path))[0]

        sample_rate = 44100
        if "sample_rate" in self.config["audio"]:
            sample_rate = self.config["audio"]["sample_rate"]

        isstereo = self.config["model"].get("stereo", True)
        device_type = (
            self.device.type
            if isinstance(self.device, torch.device)
            else torch.device(self.device).type
        )
        try:
            if device_type == "cuda" and TORCHAUDIO_GPU_ENABLED:
                mix = load_audio_tensor(
                    path, sample_rate, force_mono=not isstereo
                )
            else:
                mix = load_audio(path, sample_rate, force_mono=not isstereo)
            sr = sample_rate
        except Exception as e:
            print(i18n("无法读取音频：%s") % path)
            print(i18n("错误信息：%s") % str(e))
            return

        if isstereo:
            if mix.ndim == 1:
                mix = mix.unsqueeze(0) if torch.is_tensor(mix) else mix[np.newaxis, :]
            if mix.shape[0] == 1:
                mix = mix.repeat(2, 1) if torch.is_tensor(mix) else np.repeat(mix, 2, axis=0)
            elif mix.shape[0] > 2:
                mix = mix[:2].contiguous() if torch.is_tensor(mix) else np.ascontiguousarray(mix[:2])
        else:
            if mix.ndim == 1:
                mix = mix.unsqueeze(0) if torch.is_tensor(mix) else mix[np.newaxis, :]
            elif mix.shape[0] > 1:
                mix = (
                    mix.mean(dim=0, keepdim=True)
                    if torch.is_tensor(mix)
                    else np.mean(mix, axis=0, keepdims=True)
                )
                print(i18n("音频包含多个声道，但模型仅支持单声道，将对所有声道取平均值"))

        if torch.is_tensor(mix):
            keep_on_gpu = mix.device.type == "cuda"
            if keep_on_gpu:
                free_bytes, _ = torch.cuda.mem_get_info(mix.device)
                input_bytes = mix.numel() * mix.element_size()
                keep_on_gpu = input_bytes <= min(
                    512 * 1024 * 1024,
                    int(free_bytes * 0.10),
                )
            if keep_on_gpu:
                mixture = mix
                mix_orig = mix.detach().float().cpu().numpy()
            else:
                mixture = mix.detach().float().cpu()
                mix_orig = mixture.numpy()
                del mix
        else:
            mix = np.ascontiguousarray(mix, dtype=np.float32)
            mix_orig = mix
            mixture = torch.from_numpy(mix)
        res = self.demix_track(self.model, mixture, self.device)

        if self.config["training"]["target_instrument"] is not None:
            # if target instrument is specified, save target instrument as vocal and other instruments as others
            # other instruments are caculated by subtracting target instrument from mixture
            target_instrument = self.config["training"]["target_instrument"]
            other_instruments = [i for i in self.config["training"]["instruments"] if i != target_instrument]
            np.subtract(mix_orig, res[target_instrument], out=mix_orig)
            other = mix_orig

            path_vocal = "{}/{}_{}.wav".format(vocal_root, file_base_name, target_instrument)
            path_other = "{}/{}_{}.wav".format(others_root, file_base_name, other_instruments[0])
            self.save_audio(path_vocal, res[target_instrument].T, sr, format)
            self.save_audio(path_other, other.T, sr, format)
        else:
            # if target instrument is not specified, save the first instrument as vocal and the rest as others
            vocal_inst = self.config["training"]["instruments"][0]
            path_vocal = "{}/{}_{}.wav".format(vocal_root, file_base_name, vocal_inst)
            self.save_audio(path_vocal, res[vocal_inst].T, sr, format)
            for other in self.config["training"]["instruments"][1:]:  # save other instruments
                path_other = "{}/{}_{}.wav".format(others_root, file_base_name, other)
                self.save_audio(path_other, res[other].T, sr, format)

    def save_audio(self, path, data, sr, format):
        # input path should be endwith '.wav'
        if format in ["wav", "flac"]:
            if format == "flac":
                path = path[:-3] + "flac"
            sf.write(path, data, sr)
        else:
            sf.write(path, data, sr)
            os.system('ffmpeg -i "{}" -vn "{}" -q:a 2 -y'.format(path, path[:-3] + format))
            try:
                os.remove(path)
            except:
                pass

    def __init__(self, model_path, config_path, device, is_half):
        self.device = device
        self.is_half = is_half
        self.model_type = None
        self.config = None

        # get model_type, first try:
        if "bs_roformer" in model_path.lower() or "bsroformer" in model_path.lower():
            self.model_type = "bs_roformer"
        elif "mel_band_roformer" in model_path.lower() or "melbandroformer" in model_path.lower():
            self.model_type = "mel_band_roformer"

        if not os.path.exists(config_path):
            if self.model_type is None:
                # if model_type is still None, raise an error
                raise ValueError(
                    "Error: Unknown model type. If you are using a model without a configuration file, Ensure that your model name includes 'bs_roformer', 'bsroformer', 'mel_band_roformer', or 'melbandroformer'. Otherwise, you can manually place the model configuration file into 'tools/uvr5/uvr5w_weights' and ensure that the configuration file is named as '<model_name>.yaml' then try it again."
                )
            self.config = self.get_default_config()
        else:
            # if there is a configuration file
            self.config = self.get_config(config_path)
            if self.model_type is None:
                # if model_type is still None, second try, get model_type from the configuration file
                if "freqs_per_bands" in self.config["model"]:
                    # if freqs_per_bands in config, it's a bs_roformer model
                    self.model_type = "bs_roformer"
                else:
                    # else it's a mel_band_roformer model
                    self.model_type = "mel_band_roformer"

        print(i18n("检测到模型类型：%s") % self.model_type)
        model = self.get_model_from_config()
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)

        if is_half == False:
            self.model = model.to(device)
        else:
            self.model = model.half().to(device)

    def _path_audio_(self, input, others_root, vocal_root, format, is_hp3=False):
        self.run_folder(input, vocal_root, others_root, format)
