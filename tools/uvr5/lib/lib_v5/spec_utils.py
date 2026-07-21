import math

import librosa
import numpy as np
import torch
from infer.audio import resample_audio, resample_audio_tensor


_STFT_WINDOWS = {}


def _stft_window(n_fft, device):
    key = (n_fft, str(device))
    window = _STFT_WINDOWS.get(key)
    if window is None:
        window = torch.hann_window(
            n_fft,
            periodic=True,
            device=device,
            dtype=torch.float32,
        )
        _STFT_WINDOWS[key] = window
    return window


def _wave_to_spectrogram_torch(
    wave, hop_length, n_fft, mid_side=False, mid_side_b2=False, reverse=False
):
    wave = wave.to(dtype=torch.float32)
    if reverse:
        transformed = torch.flip(wave[:2], dims=(-1,))
    elif mid_side:
        transformed = torch.stack(
            ((wave[0] + wave[1]) / 2, wave[0] - wave[1])
        )
    elif mid_side_b2:
        transformed = torch.stack(
            (wave[1] + wave[0] * 0.5, wave[0] - wave[1] * 0.5)
        )
    else:
        transformed = wave[:2]
    return torch.stft(
        transformed,
        n_fft=n_fft,
        hop_length=hop_length,
        window=_stft_window(n_fft, transformed.device),
        center=True,
        pad_mode="constant",
        normalized=False,
        onesided=True,
        return_complex=True,
    )


def crop_center(h1, h2):
    h1_shape = h1.size()
    h2_shape = h2.size()

    if h1_shape[3] == h2_shape[3]:
        return h1
    elif h1_shape[3] < h2_shape[3]:
        raise ValueError("h1_shape[3] must be greater than h2_shape[3]")

    # s_freq = (h2_shape[2] - h1_shape[2]) // 2
    # e_freq = s_freq + h1_shape[2]
    s_time = (h1_shape[3] - h2_shape[3]) // 2
    e_time = s_time + h2_shape[3]
    h1 = h1[:, :, :, s_time:e_time]

    return h1


def wave_to_spectrogram_mt(wave, hop_length, n_fft, mid_side=False, mid_side_b2=False, reverse=False):
    if torch.is_tensor(wave):
        return _wave_to_spectrogram_torch(
            wave, hop_length, n_fft, mid_side, mid_side_b2, reverse
        )
    import threading

    if reverse:
        wave_left = np.flip(np.asfortranarray(wave[0]))
        wave_right = np.flip(np.asfortranarray(wave[1]))
    elif mid_side:
        wave_left = np.asfortranarray(np.add(wave[0], wave[1]) / 2)
        wave_right = np.asfortranarray(np.subtract(wave[0], wave[1]))
    elif mid_side_b2:
        wave_left = np.asfortranarray(np.add(wave[1], wave[0] * 0.5))
        wave_right = np.asfortranarray(np.subtract(wave[0], wave[1] * 0.5))
    else:
        wave_left = np.asfortranarray(wave[0])
        wave_right = np.asfortranarray(wave[1])

    def run_thread(**kwargs):
        global spec_left
        spec_left = librosa.stft(**kwargs)

    thread = threading.Thread(
        target=run_thread,
        kwargs={"y": wave_left, "n_fft": n_fft, "hop_length": hop_length},
    )
    thread.start()
    spec_right = librosa.stft(wave_right, n_fft=n_fft, hop_length=hop_length)
    thread.join()

    spec = np.asfortranarray([spec_left, spec_right])

    return spec


def combine_spectrograms(specs, mp):
    l = min([specs[i].shape[2] for i in specs])
    first = specs[next(iter(specs))]
    if torch.is_tensor(first):
        spec_c = torch.zeros(
            (2, mp.param["bins"] + 1, l),
            dtype=torch.complex64,
            device=first.device,
        )
    else:
        spec_c = np.zeros(shape=(2, mp.param["bins"] + 1, l), dtype=np.complex64)
    offset = 0
    bands_n = len(mp.param["band"])

    for d in range(1, bands_n + 1):
        h = mp.param["band"][d]["crop_stop"] - mp.param["band"][d]["crop_start"]
        spec_c[:, offset : offset + h, :l] = specs[d][
            :, mp.param["band"][d]["crop_start"] : mp.param["band"][d]["crop_stop"], :l
        ]
        offset += h

    if offset > mp.param["bins"]:
        raise ValueError("Too much bins")

    # lowpass fiter
    if mp.param["pre_filter_start"] > 0:  # and mp.param['band'][bands_n]['res_type'] in ['scipy', 'polyphase']:
        if bands_n == 1:
            spec_c = fft_lp_filter(spec_c, mp.param["pre_filter_start"], mp.param["pre_filter_stop"])
        else:
            gp = 1
            for b in range(mp.param["pre_filter_start"] + 1, mp.param["pre_filter_stop"]):
                g = math.pow(10, -(b - mp.param["pre_filter_start"]) * (3.5 - gp) / 20.0)
                gp = g
                spec_c[:, b, :] *= g

    if torch.is_tensor(spec_c):
        return spec_c.contiguous()
    return np.asfortranarray(spec_c)


def mask_silence(mag, ref, thres=0.2, min_range=64, fade_size=32):
    if min_range < fade_size * 2:
        raise ValueError("min_range must be >= fade_area * 2")

    if torch.is_tensor(mag):
        mag = mag.clone()
        idx = torch.where(ref.mean(dim=(0, 1)) < thres)[0]
        if idx.numel() == 0:
            return mag
        breaks = torch.where(torch.diff(idx) != 1)[0]
        starts = torch.cat((idx[:1], idx[breaks + 1]))
        ends = torch.cat((idx[breaks], idx[-1:]))
        informative = torch.where(ends - starts > min_range)[0]
        old_e = None
        for position in informative.tolist():
            s = int(starts[position].item())
            e = int(ends[position].item())
            if old_e is not None and s - old_e < fade_size:
                s = old_e - fade_size * 2
            if s != 0:
                weight = torch.linspace(
                    0,
                    1,
                    fade_size,
                    device=mag.device,
                    dtype=mag.dtype,
                )
                mag[:, :, s : s + fade_size] += (
                    weight * ref[:, :, s : s + fade_size]
                )
            else:
                s -= fade_size
            if e != mag.shape[2]:
                weight = torch.linspace(
                    1,
                    0,
                    fade_size,
                    device=mag.device,
                    dtype=mag.dtype,
                )
                mag[:, :, e - fade_size : e] += (
                    weight * ref[:, :, e - fade_size : e]
                )
            else:
                e += fade_size
            mag[:, :, s + fade_size : e - fade_size] += ref[
                :, :, s + fade_size : e - fade_size
            ]
            old_e = e
        return mag

    mag = mag.copy()

    idx = np.where(ref.mean(axis=(0, 1)) < thres)[0]
    starts = np.insert(idx[np.where(np.diff(idx) != 1)[0] + 1], 0, idx[0])
    ends = np.append(idx[np.where(np.diff(idx) != 1)[0]], idx[-1])
    uninformative = np.where(ends - starts > min_range)[0]
    if len(uninformative) > 0:
        starts = starts[uninformative]
        ends = ends[uninformative]
        old_e = None
        for s, e in zip(starts, ends):
            if old_e is not None and s - old_e < fade_size:
                s = old_e - fade_size * 2

            if s != 0:
                weight = np.linspace(0, 1, fade_size)
                mag[:, :, s : s + fade_size] += weight * ref[:, :, s : s + fade_size]
            else:
                s -= fade_size

            if e != mag.shape[2]:
                weight = np.linspace(1, 0, fade_size)
                mag[:, :, e - fade_size : e] += weight * ref[:, :, e - fade_size : e]
            else:
                e += fade_size

            mag[:, :, s + fade_size : e - fade_size] += ref[:, :, s + fade_size : e - fade_size]
            old_e = e

    return mag


def spectrogram_to_wave(spec, hop_length, mid_side, mid_side_b2, reverse):
    if torch.is_tensor(spec):
        n_fft = (spec.shape[1] - 1) * 2
        wave = torch.istft(
            spec.to(dtype=torch.complex64),
            n_fft=n_fft,
            hop_length=hop_length,
            window=_stft_window(n_fft, spec.device),
            center=True,
            normalized=False,
            onesided=True,
            return_complex=False,
        )
        wave_left, wave_right = wave[0], wave[1]
        if reverse:
            return torch.stack(
                (torch.flip(wave_left, dims=(-1,)), torch.flip(wave_right, dims=(-1,)))
            )
        if mid_side:
            return torch.stack(
                (wave_left + wave_right / 2, wave_left - wave_right / 2)
            )
        if mid_side_b2:
            return torch.stack(
                (wave_right / 1.25 + 0.4 * wave_left, wave_left / 1.25 - 0.4 * wave_right)
            )
        return wave

    spec_left = np.asfortranarray(spec[0])
    spec_right = np.asfortranarray(spec[1])

    wave_left = librosa.istft(spec_left, hop_length=hop_length)
    wave_right = librosa.istft(spec_right, hop_length=hop_length)

    if reverse:
        return np.asfortranarray([np.flip(wave_left), np.flip(wave_right)])
    elif mid_side:
        return np.asfortranarray([np.add(wave_left, wave_right / 2), np.subtract(wave_left, wave_right / 2)])
    elif mid_side_b2:
        return np.asfortranarray(
            [
                np.add(wave_right / 1.25, 0.4 * wave_left),
                np.subtract(wave_left / 1.25, 0.4 * wave_right),
            ]
        )
    else:
        return np.asfortranarray([wave_left, wave_right])


def cmb_spectrogram_to_wave(spec_m, mp, extra_bins_h=None, extra_bins=None):
    wave_band = {}
    bands_n = len(mp.param["band"])
    offset = 0

    for d in range(1, bands_n + 1):
        bp = mp.param["band"][d]
        shape = (2, bp["n_fft"] // 2 + 1, spec_m.shape[2])
        if torch.is_tensor(spec_m):
            spec_s = torch.zeros(shape, dtype=spec_m.dtype, device=spec_m.device)
        else:
            spec_s = np.ndarray(shape=shape, dtype=complex)
        h = bp["crop_stop"] - bp["crop_start"]
        spec_s[:, bp["crop_start"] : bp["crop_stop"], :] = spec_m[:, offset : offset + h, :]

        offset += h
        if d == bands_n:  # higher
            if extra_bins_h:  # if --high_end_process bypass
                max_bin = bp["n_fft"] // 2
                spec_s[:, max_bin - extra_bins_h : max_bin, :] = extra_bins[:, :extra_bins_h, :]
            if bp["hpf_start"] > 0:
                spec_s = fft_hp_filter(spec_s, bp["hpf_start"], bp["hpf_stop"] - 1)
            if bands_n == 1:
                wave = spectrogram_to_wave(
                    spec_s,
                    bp["hl"],
                    mp.param["mid_side"],
                    mp.param["mid_side_b2"],
                    mp.param["reverse"],
                )
            else:
                wave = wave + spectrogram_to_wave(
                        spec_s,
                        bp["hl"],
                        mp.param["mid_side"],
                        mp.param["mid_side_b2"],
                        mp.param["reverse"],
                    )
        else:
            sr = mp.param["band"][d + 1]["sr"]
            if d == 1:  # lower
                spec_s = fft_lp_filter(spec_s, bp["lpf_start"], bp["lpf_stop"])
                band_wave = spectrogram_to_wave(
                        spec_s,
                        bp["hl"],
                        mp.param["mid_side"],
                        mp.param["mid_side_b2"],
                        mp.param["reverse"],
                    )
                if torch.is_tensor(band_wave):
                    wave = resample_audio_tensor(
                        band_wave, bp["sr"], sr, force_mono=False
                    )
                else:
                    wave = resample_audio(
                        band_wave,
                        bp["sr"],
                        sr,
                        force_mono=False,
                        res_type="sinc_fastest",
                    )
            else:  # mid
                spec_s = fft_hp_filter(spec_s, bp["hpf_start"], bp["hpf_stop"] - 1)
                spec_s = fft_lp_filter(spec_s, bp["lpf_start"], bp["lpf_stop"])
                wave2 = wave + spectrogram_to_wave(
                        spec_s,
                        bp["hl"],
                        mp.param["mid_side"],
                        mp.param["mid_side_b2"],
                        mp.param["reverse"],
                    )
                if torch.is_tensor(wave2):
                    wave = resample_audio_tensor(
                        wave2, bp["sr"], sr, force_mono=False
                    )
                else:
                    wave = resample_audio(
                        wave2,
                        bp["sr"],
                        sr,
                        force_mono=False,
                        res_type="scipy",
                    )

    return wave.transpose(0, 1) if torch.is_tensor(wave) else wave.T


def fft_lp_filter(spec, bin_start, bin_stop):
    g = 1.0
    for b in range(bin_start, bin_stop):
        g -= 1 / (bin_stop - bin_start)
        spec[:, b, :] = g * spec[:, b, :]

    spec[:, bin_stop:, :] *= 0

    return spec


def fft_hp_filter(spec, bin_start, bin_stop):
    g = 1.0
    for b in range(bin_start, bin_stop, -1):
        g -= 1 / (bin_start - bin_stop)
        spec[:, b, :] = g * spec[:, b, :]

    spec[:, 0 : bin_stop + 1, :] *= 0

    return spec


def mirroring(a, spec_m, input_high_end, mp):
    if torch.is_tensor(spec_m):
        source = spec_m[
            :,
            mp.param["pre_filter_start"]
            - 10
            - input_high_end.shape[1] : mp.param["pre_filter_start"]
            - 10,
            :,
        ]
        mirror = torch.flip(torch.abs(source), dims=(1,))
        if "mirroring" == a:
            mirror = torch.polar(mirror, torch.angle(input_high_end))
            return torch.where(
                torch.abs(input_high_end) <= torch.abs(mirror),
                input_high_end,
                mirror,
            )
        if "mirroring2" == a:
            mirror = mirror * input_high_end * 1.7
            return torch.where(
                torch.abs(input_high_end) <= torch.abs(mirror),
                input_high_end,
                mirror,
            )

    if "mirroring" == a:
        mirror = np.flip(
            np.abs(
                spec_m[
                    :,
                    mp.param["pre_filter_start"] - 10 - input_high_end.shape[1] : mp.param["pre_filter_start"] - 10,
                    :,
                ]
            ),
            1,
        )
        mirror = mirror * np.exp(1.0j * np.angle(input_high_end))

        return np.where(np.abs(input_high_end) <= np.abs(mirror), input_high_end, mirror)

    if "mirroring2" == a:
        mirror = np.flip(
            np.abs(
                spec_m[
                    :,
                    mp.param["pre_filter_start"] - 10 - input_high_end.shape[1] : mp.param["pre_filter_start"] - 10,
                    :,
                ]
            ),
            1,
        )
        mi = np.multiply(mirror, input_high_end * 1.7)

        return np.where(np.abs(input_high_end) <= np.abs(mi), input_high_end, mi)
