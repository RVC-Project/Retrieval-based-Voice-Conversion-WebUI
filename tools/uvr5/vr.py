import os

parent_directory = os.path.dirname(os.path.abspath(__file__))
import logging

logger = logging.getLogger(__name__)

import numpy as np
import soundfile as sf
import torch
from infer.audio import (
    TORCHAUDIO_GPU_ENABLED,
    load_audio,
    load_audio_tensor,
    resample_audio,
    resample_audio_tensor,
)
from tools.uvr5.lib.lib_v5 import nets_61968KB as Nets
from tools.uvr5.lib.lib_v5 import spec_utils
from tools.uvr5.lib.lib_v5.model_param_init import ModelParameters
from tools.uvr5.lib.lib_v5.nets_new import CascadedNet
from tools.uvr5.lib.utils import inference


def _ensure_stereo(audio):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    if audio.shape[0] == 1:
        return np.repeat(audio, 2, axis=0)
    if audio.shape[0] > 2:
        return np.ascontiguousarray(audio[:2])
    return audio


def _ensure_stereo_tensor(audio, device):
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    return audio.to(device=device)


def _cuda_device(device):
    parsed = device if isinstance(device, torch.device) else torch.device(device)
    return parsed if parsed.type == "cuda" else None


def _vr_gpu_memory_fits(audio, mp, device):
    highest_band = len(mp.param["band"])
    frames = max(
        1,
        int(audio.shape[-1] // mp.param["band"][highest_band]["hl"] + 1),
    )
    band_bins = sum(
        mp.param["band"][band]["n_fft"] // 2 + 1
        for band in mp.param["band"]
    )
    combined_bins = mp.param["bins"] + 1
    # Complex band spectra + combined/target spectra + magnitude/prediction.
    estimated = frames * 2 * (
        band_bins * 8 + combined_bins * (8 * 3 + 4 * 3)
    )
    free_bytes, _ = torch.cuda.mem_get_info(device)
    return estimated <= int(free_bytes * 0.42)


def _prepare_spectrogram(music_file, mp, data, device, allow_gpu=True):
    cuda_device = _cuda_device(device)
    use_gpu = bool(
        allow_gpu and cuda_device is not None and TORCHAUDIO_GPU_ENABLED
    )
    if use_gpu:
        try:
            high_sr = mp.param["band"][len(mp.param["band"])]["sr"]
            high_wave = _ensure_stereo_tensor(
                load_audio_tensor(music_file, high_sr, force_mono=False),
                cuda_device,
            )
            if not _vr_gpu_memory_fits(high_wave, mp, cuda_device):
                use_gpu = False
                high_wave = high_wave.float().cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            use_gpu = False
            high_wave = None
    else:
        high_wave = None

    input_high_end_h = None
    input_high_end = None
    X_spec_s = {}
    bands_n = len(mp.param["band"])
    previous_wave = None
    for d in range(bands_n, 0, -1):
        bp = mp.param["band"][d]
        if d == bands_n:
            if high_wave is None:
                current_wave = _ensure_stereo(
                    load_audio(music_file, bp["sr"], force_mono=False)
                )
            else:
                current_wave = high_wave
        elif use_gpu:
            current_wave = resample_audio_tensor(
                previous_wave,
                mp.param["band"][d + 1]["sr"],
                bp["sr"],
                force_mono=False,
            )
        else:
            current_wave = resample_audio(
                previous_wave,
                mp.param["band"][d + 1]["sr"],
                bp["sr"],
                force_mono=False,
                res_type=bp["res_type"],
            )
        X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(
            current_wave,
            bp["hl"],
            bp["n_fft"],
            mp.param["mid_side"],
            mp.param["mid_side_b2"],
            mp.param["reverse"],
        )
        if d == bands_n and data["high_end_process"] != "none":
            input_high_end_h = (bp["n_fft"] // 2 - bp["crop_stop"]) + (
                mp.param["pre_filter_stop"] - mp.param["pre_filter_start"]
            )
            input_high_end = X_spec_s[d][
                :, bp["n_fft"] // 2 - input_high_end_h : bp["n_fft"] // 2, :
            ]
            if torch.is_tensor(input_high_end):
                input_high_end = input_high_end.clone()
        previous_wave = current_wave

    X_spec_m = spec_utils.combine_spectrograms(X_spec_s, mp)
    del previous_wave, X_spec_s
    return X_spec_m, input_high_end_h, input_high_end


def _wave_for_write(wave):
    if torch.is_tensor(wave):
        return wave.detach().to(device="cpu", dtype=torch.float32).numpy()
    return np.asarray(wave)


def _separate_spectrogram(X_spec_m, device, model, aggressiveness, data):
    with torch.no_grad():
        pred, X_mag, X_phase = inference(
            X_spec_m, device, model, aggressiveness, data
        )
    if data["postprocess"]:
        if torch.is_tensor(pred):
            pred_inv = torch.clamp(X_mag - pred, min=0)
        else:
            pred_inv = np.clip(X_mag - pred, 0, np.inf)
        pred = spec_utils.mask_silence(pred, pred_inv)
    if torch.is_tensor(X_spec_m):
        ratio = pred.float() / X_mag.clamp_min(1e-8)
        ratio = torch.nan_to_num(ratio)
        y_spec_m = X_spec_m * ratio
    else:
        y_spec_m = pred * X_phase
    return y_spec_m


class AudioPre:
    def __init__(self, agg, model_path, device, is_half, tta=False):
        self.model_path = model_path
        self.device = device
        self.data = {
            # Processing Options
            "postprocess": False,
            "tta": tta,
            # Constants
            "window_size": 512,
            "agg": agg,
            "high_end_process": "mirroring",
        }
        mp = ModelParameters("%s/lib/lib_v5/modelparams/4band_v2.json" % parent_directory)
        model = Nets.CascadedASPPNet(mp.param["bins"] * 2)
        cpk = torch.load(model_path, map_location="cpu")
        model.load_state_dict(cpk)
        model.eval()
        if is_half:
            model = model.half().to(device)
        else:
            model = model.to(device)

        self.mp = mp
        self.model = model

    def _path_audio_(self, music_file, ins_root=None, vocal_root=None, format="flac", is_hp3=False):
        if ins_root is None and vocal_root is None:
            return "No save root."
        name = os.path.basename(music_file)
        if ins_root is not None:
            os.makedirs(ins_root, exist_ok=True)
        if vocal_root is not None:
            os.makedirs(vocal_root, exist_ok=True)
        aggresive_set = float(self.data["agg"] / 100)
        aggressiveness = {
            "value": aggresive_set,
            "split_bin": self.mp.param["band"][1]["crop_stop"],
        }
        gpu_oom = False
        try:
            X_spec_m, input_high_end_h, input_high_end = _prepare_spectrogram(
                music_file, self.mp, self.data, self.device
            )
            y_spec_m = _separate_spectrogram(
                X_spec_m, self.device, self.model, aggressiveness, self.data
            )
        except torch.cuda.OutOfMemoryError:
            X_spec_m = None
            input_high_end = None
            y_spec_m = None
            gpu_oom = True
        if gpu_oom:
            torch.cuda.empty_cache()
            X_spec_m, input_high_end_h, input_high_end = _prepare_spectrogram(
                music_file, self.mp, self.data, self.device, allow_gpu=False
            )
            y_spec_m = _separate_spectrogram(
                X_spec_m, self.device, self.model, aggressiveness, self.data
            )

        if is_hp3 == True:
            ins_root, vocal_root = vocal_root, ins_root

        if ins_root is not None:
            if self.data["high_end_process"].startswith("mirroring"):
                input_high_end_ = spec_utils.mirroring(self.data["high_end_process"], y_spec_m, input_high_end, self.mp)
                wav_instrument = spec_utils.cmb_spectrogram_to_wave(
                    y_spec_m, self.mp, input_high_end_h, input_high_end_
                )
            else:
                wav_instrument = spec_utils.cmb_spectrogram_to_wave(y_spec_m, self.mp)
            logger.info("%s instruments done" % name)
            if is_hp3 == True:
                head = "vocal_"
            else:
                head = "instrument_"
            if format in ["wav", "flac"]:
                sf.write(
                    os.path.join(
                        ins_root,
                        head + "{}_{}.{}".format(name, self.data["agg"], format),
                    ),
                    (_wave_for_write(wav_instrument) * 32768).astype("int16"),
                    self.mp.param["sr"],
                )  #
            else:
                path = os.path.join(ins_root, head + "{}_{}.wav".format(name, self.data["agg"]))
                sf.write(
                    path,
                    (_wave_for_write(wav_instrument) * 32768).astype("int16"),
                    self.mp.param["sr"],
                )
                if os.path.exists(path):
                    opt_format_path = path[:-4] + ".%s" % format
                    cmd = 'ffmpeg -i "%s" -vn "%s" -q:a 2 -y' % (path, opt_format_path)
                    print(cmd)
                    os.system(cmd)
                    if os.path.exists(opt_format_path):
                        try:
                            os.remove(path)
                        except:
                            pass
        if vocal_root is not None:
            if torch.is_tensor(y_spec_m):
                y_spec_m.neg_().add_(X_spec_m)
                v_spec_m = y_spec_m
            else:
                np.subtract(X_spec_m, y_spec_m, out=y_spec_m)
                v_spec_m = y_spec_m
            if is_hp3 == True:
                head = "instrument_"
            else:
                head = "vocal_"
            if self.data["high_end_process"].startswith("mirroring"):
                input_high_end_ = spec_utils.mirroring(self.data["high_end_process"], v_spec_m, input_high_end, self.mp)
                wav_vocals = spec_utils.cmb_spectrogram_to_wave(v_spec_m, self.mp, input_high_end_h, input_high_end_)
            else:
                wav_vocals = spec_utils.cmb_spectrogram_to_wave(v_spec_m, self.mp)
            logger.info("%s vocals done" % name)
            if format in ["wav", "flac"]:
                sf.write(
                    os.path.join(
                        vocal_root,
                        head + "{}_{}.{}".format(name, self.data["agg"], format),
                    ),
                    (_wave_for_write(wav_vocals) * 32768).astype("int16"),
                    self.mp.param["sr"],
                )
            else:
                path = os.path.join(vocal_root, head + "{}_{}.wav".format(name, self.data["agg"]))
                sf.write(
                    path,
                    (_wave_for_write(wav_vocals) * 32768).astype("int16"),
                    self.mp.param["sr"],
                )
                if os.path.exists(path):
                    opt_format_path = path[:-4] + ".%s" % format
                    cmd = 'ffmpeg -i "%s" -vn "%s" -q:a 2 -y' % (path, opt_format_path)
                    print(cmd)
                    os.system(cmd)
                    if os.path.exists(opt_format_path):
                        try:
                            os.remove(path)
                        except:
                            pass


class AudioPreDeEcho:
    def __init__(self, agg, model_path, device, is_half, tta=False):
        self.model_path = model_path
        self.device = device
        self.data = {
            # Processing Options
            "postprocess": False,
            "tta": tta,
            # Constants
            "window_size": 512,
            "agg": agg,
            "high_end_process": "mirroring",
        }
        mp = ModelParameters("%s/lib/lib_v5/modelparams/4band_v3.json" % parent_directory)
        nout = 64 if "DeReverb" in model_path else 48
        model = CascadedNet(mp.param["bins"] * 2, nout)
        cpk = torch.load(model_path, map_location="cpu")
        model.load_state_dict(cpk)
        model.eval()
        if is_half:
            model = model.half().to(device)
        else:
            model = model.to(device)

        self.mp = mp
        self.model = model

    def _path_audio_(
        self, music_file, vocal_root=None, ins_root=None, format="flac", is_hp3=False
    ):  # 3个VR模型vocal和ins是反的
        if ins_root is None and vocal_root is None:
            return "No save root."
        name = os.path.basename(music_file)
        if ins_root is not None:
            os.makedirs(ins_root, exist_ok=True)
        if vocal_root is not None:
            os.makedirs(vocal_root, exist_ok=True)
        aggresive_set = float(self.data["agg"] / 100)
        aggressiveness = {
            "value": aggresive_set,
            "split_bin": self.mp.param["band"][1]["crop_stop"],
        }
        gpu_oom = False
        try:
            X_spec_m, input_high_end_h, input_high_end = _prepare_spectrogram(
                music_file, self.mp, self.data, self.device
            )
            y_spec_m = _separate_spectrogram(
                X_spec_m, self.device, self.model, aggressiveness, self.data
            )
        except torch.cuda.OutOfMemoryError:
            X_spec_m = None
            input_high_end = None
            y_spec_m = None
            gpu_oom = True
        if gpu_oom:
            torch.cuda.empty_cache()
            X_spec_m, input_high_end_h, input_high_end = _prepare_spectrogram(
                music_file, self.mp, self.data, self.device, allow_gpu=False
            )
            y_spec_m = _separate_spectrogram(
                X_spec_m, self.device, self.model, aggressiveness, self.data
            )

        if ins_root is not None:
            if self.data["high_end_process"].startswith("mirroring"):
                input_high_end_ = spec_utils.mirroring(self.data["high_end_process"], y_spec_m, input_high_end, self.mp)
                wav_instrument = spec_utils.cmb_spectrogram_to_wave(
                    y_spec_m, self.mp, input_high_end_h, input_high_end_
                )
            else:
                wav_instrument = spec_utils.cmb_spectrogram_to_wave(y_spec_m, self.mp)
            logger.info("%s instruments done" % name)
            if format in ["wav", "flac"]:
                sf.write(
                    os.path.join(
                        ins_root,
                        "vocal_{}_{}.{}".format(name, self.data["agg"], format),
                    ),
                    (_wave_for_write(wav_instrument) * 32768).astype("int16"),
                    self.mp.param["sr"],
                )  #
            else:
                path = os.path.join(ins_root, "vocal_{}_{}.wav".format(name, self.data["agg"]))
                sf.write(
                    path,
                    (_wave_for_write(wav_instrument) * 32768).astype("int16"),
                    self.mp.param["sr"],
                )
                if os.path.exists(path):
                    opt_format_path = path[:-4] + ".%s" % format
                    cmd = 'ffmpeg -i "%s" -vn "%s" -q:a 2 -y' % (path, opt_format_path)
                    print(cmd)
                    os.system(cmd)
                    if os.path.exists(opt_format_path):
                        try:
                            os.remove(path)
                        except:
                            pass
        if vocal_root is not None:
            if torch.is_tensor(y_spec_m):
                y_spec_m.neg_().add_(X_spec_m)
                v_spec_m = y_spec_m
            else:
                np.subtract(X_spec_m, y_spec_m, out=y_spec_m)
                v_spec_m = y_spec_m
            if self.data["high_end_process"].startswith("mirroring"):
                input_high_end_ = spec_utils.mirroring(self.data["high_end_process"], v_spec_m, input_high_end, self.mp)
                wav_vocals = spec_utils.cmb_spectrogram_to_wave(v_spec_m, self.mp, input_high_end_h, input_high_end_)
            else:
                wav_vocals = spec_utils.cmb_spectrogram_to_wave(v_spec_m, self.mp)
            logger.info("%s vocals done" % name)
            if format in ["wav", "flac"]:
                sf.write(
                    os.path.join(
                        vocal_root,
                        "instrument_{}_{}.{}".format(name, self.data["agg"], format),
                    ),
                    (_wave_for_write(wav_vocals) * 32768).astype("int16"),
                    self.mp.param["sr"],
                )
            else:
                path = os.path.join(vocal_root, "instrument_{}_{}.wav".format(name, self.data["agg"]))
                sf.write(
                    path,
                    (_wave_for_write(wav_vocals) * 32768).astype("int16"),
                    self.mp.param["sr"],
                )
                if os.path.exists(path):
                    opt_format_path = path[:-4] + ".%s" % format
                    cmd = 'ffmpeg -i "%s" -vn "%s" -q:a 2 -y' % (path, opt_format_path)
                    print(cmd)
                    os.system(cmd)
                    if os.path.exists(opt_format_path):
                        try:
                            os.remove(path)
                        except:
                            pass
