import math
import platform
import traceback

import librosa
import numpy as np
import torch


ARM = "arm"
wav_resolution = (
    "polyphase" if platform.system() == "Darwin" and (platform.processor() == ARM or ARM in platform.platform()) else "soxr_hq"
)


_HANN_WINDOW_CACHE = {}
_FILTER_MASK_CACHE = {}


def _hann_window(n_fft, dtype, device):
    torch_device = torch.device(device)
    key = (int(n_fft), dtype, torch_device.type, torch_device.index)
    window = _HANN_WINDOW_CACHE.get(key)
    if window is None:
        window = torch.hann_window(n_fft, dtype=dtype, device=torch_device)
        _HANN_WINDOW_CACHE[key] = window
    return window


def resample_audio(wave, orig_sr, target_sr, res_type=None):
    orig_sr = int(orig_sr)
    target_sr = int(target_sr)
    if orig_sr == target_sr:
        return np.asfortranarray(wave)

    last_error = None
    for candidate in dict.fromkeys(([res_type] if res_type else []) + ["soxr_hq", "polyphase"]):
        try:
            return librosa.resample(wave, orig_sr=orig_sr, target_sr=target_sr, res_type=candidate)
        except (ImportError, ModuleNotFoundError) as exc:
            last_error = exc

    try:
        return _linear_resample(wave, orig_sr, target_sr)
    except Exception:
        if last_error is not None:
            raise last_error
        raise


def _linear_resample(wave, orig_sr, target_sr):
    wave = np.asarray(wave)
    original_length = wave.shape[-1]
    target_length = max(1, int(round(original_length * target_sr / orig_sr)))
    if original_length == target_length:
        return np.asfortranarray(wave)

    old_x, new_x = np.linspace(0.0, 1.0, original_length, endpoint=False), np.linspace(0.0, 1.0, target_length, endpoint=False)
    if wave.ndim == 1:
        return np.asfortranarray(np.interp(new_x, old_x, wave).astype(wave.dtype, copy=False))
    return np.asfortranarray(
        np.stack(
            [
                np.interp(new_x, old_x, channel).astype(wave.dtype, copy=False)
                for channel in wave.reshape((-1, original_length))
            ],
            axis=0,
        ).reshape(wave.shape[:-1] + (target_length,))
    )


def crop_center(h1, h2):
    h1_time, h2_time = h1.size(3), h2.size(3)
    if h1_time == h2_time:
        return h1
    if h1_time < h2_time:
        raise ValueError("h1_shape[3] must be greater than h2_shape[3]")
    start = (h1_time - h2_time) // 2
    return h1[:, :, :, start : start + h2_time]


def preprocess(x_spec):
    return np.abs(x_spec), np.angle(x_spec)


def make_padding(width, cropsize, offset):
    roi_size = cropsize - offset * 2 or cropsize
    return offset, roi_size - (width % roi_size) + offset, roi_size


def merge_artifacts(y_mask, thres=0.01, min_range=64, fade_size=32):
    mask = y_mask
    try:
        if min_range < fade_size * 2:
            raise ValueError("min_range must be >= fade_size * 2")

        idx = np.where(y_mask.min(axis=(0, 1)) > thres)[0]
        if len(idx) == 0:
            return mask
        start_idx = np.insert(idx[np.where(np.diff(idx) != 1)[0] + 1], 0, idx[0])
        end_idx = np.append(idx[np.where(np.diff(idx) != 1)[0]], idx[-1])
        artifact_idx = np.where(end_idx - start_idx > min_range)[0]
        weight = np.zeros_like(y_mask)
        if len(artifact_idx) > 0:
            start_idx = start_idx[artifact_idx]
            end_idx = end_idx[artifact_idx]
            old_e = None
            for s, e in zip(start_idx, end_idx):
                if old_e is not None and s - old_e < fade_size:
                    s = old_e - fade_size * 2
                if s != 0:
                    weight[:, :, s : s + fade_size] = np.linspace(0, 1, fade_size)
                else:
                    s -= fade_size
                if e != y_mask.shape[2]:
                    weight[:, :, e - fade_size : e] = np.linspace(1, 0, fade_size)
                else:
                    e += fade_size
                weight[:, :, s + fade_size : e - fade_size] = 1
                old_e = e

        y_mask += weight * (1 - y_mask)
        mask = y_mask
    except Exception as exc:
        print(f'Post Process Failed: {type(exc).__name__}: "{exc}"\n{"".join(traceback.format_tb(exc.__traceback__))}"')
    return mask


def convert_channels(spec, mp, band):
    mode = mp.param["band"][band].get("convert_channels")
    if mode == "mid_side_c":
        return np.asfortranarray([np.add(spec[0], spec[1] * 0.25), np.subtract(spec[1], spec[0] * 0.25)])
    if mode == "mid_side":
        return np.asfortranarray([np.add(spec[0], spec[1]) / 2, np.subtract(spec[0], spec[1])])
    if mode == "stereo_n":
        return np.asfortranarray([np.add(spec[0], spec[1] * 0.25) / 0.9375, np.add(spec[1], spec[0] * 0.25) / 0.9375])
    return spec


def combine_spectrograms(specs, mp, is_v51_model=False):
    length = min(specs[i].shape[2] for i in specs)
    spec_c = np.zeros((2, mp.param["bins"] + 1, length), dtype=np.complex64)
    offset = 0
    bands_n = len(mp.param["band"])
    pre_start, pre_stop = mp.param["pre_filter_start"], mp.param["pre_filter_stop"]

    for d in range(1, bands_n + 1):
        band = mp.param["band"][d]
        height = band["crop_stop"] - band["crop_start"]
        spec_c[:, offset : offset + height, :length] = specs[d][:, band["crop_start"] : band["crop_stop"], :length]
        offset += height

    if offset > mp.param["bins"]:
        raise ValueError("Too much bins")

    if pre_start > 0:
        if is_v51_model:
            spec_c *= get_lp_filter_mask(spec_c.shape[1], pre_start, pre_stop)
        elif bands_n == 1:
            spec_c = fft_lp_filter(spec_c, pre_start, pre_stop)
        else:
            gain_prev = 1
            for b in range(pre_start + 1, pre_stop):
                gain = math.pow(10, -(b - pre_start) * (3.5 - gain_prev) / 20.0)
                gain_prev = gain
                spec_c[:, b, :] *= gain

    return np.asfortranarray(spec_c)


def wave_to_spectrogram(wave, hop_length, n_fft, mp, band, is_v51_model=False, torch_device=None):
    if wave.ndim == 1:
        wave = np.asfortranarray([wave, wave])

    left, right = (np.asfortranarray(channel) for channel in wave[:2])
    if not is_v51_model:
        if mp.param["reverse"]:
            left, right = np.flip(left), np.flip(right)
        elif mp.param["mid_side"]:
            left, right = np.asfortranarray(np.add(wave[0], wave[1]) / 2), np.asfortranarray(np.subtract(wave[0], wave[1]))
        elif mp.param["mid_side_b2"]:
            left, right = (
                np.asfortranarray(np.add(wave[1], wave[0] * 0.5)),
                np.asfortranarray(np.subtract(wave[0], wave[1] * 0.5)),
            )

    spec = _torch_stft(np.asfortranarray([left, right]), n_fft, hop_length, torch_device)
    if spec is None:
        spec = np.asfortranarray(
            [
                librosa.stft(left, n_fft=n_fft, hop_length=hop_length),
                librosa.stft(right, n_fft=n_fft, hop_length=hop_length),
            ]
        )
    return convert_channels(spec, mp, band) if is_v51_model else spec


def _torch_stft(wave, n_fft, hop_length, device):
    if device is None or torch.device(device).type != "cuda":
        return None
    wave_t = torch.from_numpy(np.ascontiguousarray(wave)).to(device)
    window = _hann_window(n_fft, wave_t.dtype, device)
    spec = torch.stft(
        wave_t,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=True,
        pad_mode="constant",
        return_complex=True,
    )
    return np.asfortranarray(spec.cpu().numpy())


def _torch_istft(spec, hop_length, device):
    if device is None or torch.device(device).type != "cuda":
        return None
    n_fft = (spec.shape[1] - 1) * 2
    spec_t = torch.from_numpy(np.ascontiguousarray(spec)).to(device)
    window = _hann_window(n_fft, spec_t.real.dtype, device)
    wave = torch.istft(spec_t, n_fft=n_fft, hop_length=hop_length, window=window, center=True, return_complex=False)
    return np.asfortranarray(wave.cpu().numpy())


def spectrogram_to_wave(spec, hop_length=1024, mp=None, band=0, is_v51_model=True, torch_device=None):
    wave = _torch_istft(spec, hop_length, torch_device)
    if wave is None:
        left = librosa.istft(np.asfortranarray(spec[0]), hop_length=hop_length, dtype=np.float64)
        right = librosa.istft(np.asfortranarray(spec[1]), hop_length=hop_length, dtype=np.float64)
    else:
        left, right = wave[0], wave[1]

    if is_v51_model:
        mode = mp.param["band"][band].get("convert_channels")
        if mode == "mid_side_c":
            return np.asfortranarray([np.subtract(left / 1.0625, right / 4.25), np.add(right / 1.0625, left / 4.25)])
        if mode == "mid_side":
            return np.asfortranarray([np.add(left, right / 2), np.subtract(left, right / 2)])
        if mode == "stereo_n":
            return np.asfortranarray([np.subtract(left, right * 0.25), np.subtract(right, left * 0.25)])
    else:
        if mp.param["reverse"]:
            return np.asfortranarray([np.flip(left), np.flip(right)])
        if mp.param["mid_side"]:
            return np.asfortranarray([np.add(left, right / 2), np.subtract(left, right / 2)])
        if mp.param["mid_side_b2"]:
            return np.asfortranarray([np.add(right / 1.25, 0.4 * left), np.subtract(left / 1.25, 0.4 * right)])

    return np.asfortranarray([left, right])


def cmb_spectrogram_to_wave(spec_m, mp, extra_bins_h=None, extra_bins=None, is_v51_model=False, torch_device=None):
    spec_m = np.where(np.isnan(spec_m), 0, spec_m)
    extra_bins_h = None if extra_bins_h is None else int(extra_bins_h)
    extra_bins = None if extra_bins is None else np.where(np.isnan(extra_bins), 0, extra_bins)

    bands_n = len(mp.param["band"])
    offset = 0
    wave = None

    for d in range(1, bands_n + 1):
        bp = mp.param["band"][d]
        spec_s = np.zeros((2, bp["n_fft"] // 2 + 1, spec_m.shape[2]), dtype=np.result_type(spec_m.dtype, np.complex64))
        height = bp["crop_stop"] - bp["crop_start"]
        spec_s[:, bp["crop_start"] : bp["crop_stop"], :] = spec_m[:, offset : offset + height, :]
        offset += height

        if d == bands_n:
            if extra_bins_h is not None:
                spec_s[:, bp["n_fft"] // 2 - extra_bins_h : bp["n_fft"] // 2, :] = extra_bins[:, :extra_bins_h, :]
            if bp["hpf_start"] > 0:
                spec_s = (
                    spec_s * get_hp_filter_mask(spec_s.shape[1], bp["hpf_start"], bp["hpf_stop"] - 1)
                    if is_v51_model
                    else fft_hp_filter(spec_s, bp["hpf_start"], bp["hpf_stop"] - 1)
                )
            band_wave = spectrogram_to_wave(spec_s, bp["hl"], mp, d, is_v51_model, torch_device=torch_device)
            wave = band_wave if wave is None else np.add(wave, band_wave)
        else:
            sr = mp.param["band"][d + 1]["sr"]
            if d == 1:
                spec_s = (
                    spec_s * get_lp_filter_mask(spec_s.shape[1], bp["lpf_start"], bp["lpf_stop"])
                    if is_v51_model
                    else fft_lp_filter(spec_s, bp["lpf_start"], bp["lpf_stop"])
                )
                wave = resample_audio(
                    spectrogram_to_wave(spec_s, bp["hl"], mp, d, is_v51_model, torch_device=torch_device),
                    orig_sr=bp["sr"],
                    target_sr=sr,
                    res_type=wav_resolution,
                )
            else:
                if is_v51_model:
                    spec_s *= get_hp_filter_mask(spec_s.shape[1], bp["hpf_start"], bp["hpf_stop"] - 1)
                    spec_s *= get_lp_filter_mask(spec_s.shape[1], bp["lpf_start"], bp["lpf_stop"])
                else:
                    spec_s = fft_hp_filter(spec_s, bp["hpf_start"], bp["hpf_stop"] - 1)
                    spec_s = fft_lp_filter(spec_s, bp["lpf_start"], bp["lpf_stop"])
                wave = resample_audio(
                    np.add(wave, spectrogram_to_wave(spec_s, bp["hl"], mp, d, is_v51_model, torch_device=torch_device)),
                    orig_sr=bp["sr"],
                    target_sr=sr,
                    res_type=wav_resolution,
                )

    return wave


def _get_filter_mask(kind, n_bins, bin_start, bin_stop):
    key = (kind, int(n_bins), int(bin_start), int(bin_stop))
    mask = _FILTER_MASK_CACHE.get(key)
    if mask is None:
        mask = np.concatenate(
            (
                [
                    np.ones((bin_start - 1, 1)),
                    np.linspace(1, 0, bin_stop - bin_start + 1)[:, None],
                    np.zeros((n_bins - bin_stop, 1)),
                ]
                if kind == "lp"
                else [
                    np.zeros((bin_stop + 1, 1)),
                    np.linspace(0, 1, 1 + bin_start - bin_stop)[:, None],
                    np.ones((n_bins - bin_start - 2, 1)),
                ]
            ),
            axis=0,
        )
        _FILTER_MASK_CACHE[key] = mask
    return mask


def get_lp_filter_mask(n_bins, bin_start, bin_stop):
    return _get_filter_mask("lp", n_bins, bin_start, bin_stop)


def get_hp_filter_mask(n_bins, bin_start, bin_stop):
    return _get_filter_mask("hp", n_bins, bin_start, bin_stop)


def fft_lp_filter(spec, bin_start, bin_stop):
    gain = 1.0
    for b in range(bin_start, bin_stop):
        gain -= 1 / (bin_stop - bin_start)
        spec[:, b, :] = gain * spec[:, b, :]
    spec[:, bin_stop:, :] *= 0
    return spec


def fft_hp_filter(spec, bin_start, bin_stop):
    gain = 1.0
    for b in range(bin_start, bin_stop, -1):
        gain -= 1 / (bin_start - bin_stop)
        spec[:, b, :] = gain * spec[:, b, :]
    spec[:, 0 : bin_stop + 1, :] *= 0
    return spec


def mirroring(mode, spec_m, input_high_end, mp):
    if mode not in ("mirroring", "mirroring2"):
        return input_high_end
    mirror = np.flip(
        np.abs(spec_m[:, mp.param["pre_filter_start"] - 10 - input_high_end.shape[1] : mp.param["pre_filter_start"] - 10, :]), 1
    )
    mirror = (
        mirror * np.exp(1.0j * np.angle(input_high_end)) if mode == "mirroring" else np.multiply(mirror, input_high_end * 1.7)
    )
    return np.where(np.abs(input_high_end) <= np.abs(mirror), input_high_end, mirror)


def adjust_aggr(mask, is_non_accom_stem, aggressiveness):
    aggr = aggressiveness["value"] * 2
    if aggr != 0:
        if is_non_accom_stem:
            aggr = 1 - aggr
        if np.any(aggr > 10) or np.any(aggr < -10):
            print(f"Warning: Extreme aggressiveness values detected: {aggr}")

        aggr = np.array([aggr, aggr])
        if (correction := aggressiveness["aggr_correction"]) is not None:
            aggr[0] += correction["left"]
            aggr[1] += correction["right"]

        split_bin = aggressiveness["split_bin"]
        mask[:, :split_bin] = np.power(mask[:, :split_bin], 1 + aggr[:, None, None] / 3)
        mask[:, split_bin:] = np.power(mask[:, split_bin:], 1 + aggr[:, None, None])
    return mask
