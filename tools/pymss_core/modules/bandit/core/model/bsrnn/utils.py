import os
from abc import abstractmethod
from typing import Callable

import numpy as np
import torch
from torch import Tensor

from ....._dsp import hz_to_midi, mel_filterbank as _mel_filterbank, midi_to_hz


def band_widths_from_specs(band_specs):
    return [e - i for i, e in band_specs]


def check_nonzero_bandwidth(band_specs):
    for fstart, fend in band_specs:
        if fend - fstart <= 0:
            raise ValueError("Bands cannot be zero-width")


def check_no_overlap(band_specs):
    fend_prev = -1
    for fstart_curr, fend_curr in band_specs:
        if fstart_curr <= fend_prev:
            raise ValueError("Bands cannot overlap")


def check_no_gap(band_specs):
    fstart, _ = band_specs[0]
    assert fstart == 0

    fend_prev = -1
    for fstart_curr, fend_curr in band_specs:
        if fstart_curr - fend_prev > 1:
            raise ValueError("Bands cannot leave gap")
        fend_prev = fend_curr


def create_triangular_filterbank(all_freqs, f_pts):
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)
    down_slopes = -slopes[:, :-2] / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]
    return torch.clamp(torch.minimum(down_slopes, up_slopes), min=0.0)


def triangular_filterbank_from_points(all_freqs, f_pts):
    fb = create_triangular_filterbank(all_freqs, f_pts).T
    first_active_band = torch.nonzero(torch.sum(fb, dim=-1))[0, 0]
    fb[first_active_band, : torch.nonzero(fb[first_active_band, :])[0, 0]] = 1.0
    return fb


def hz_to_bark(hz):
    return 6 * np.arcsinh(np.asarray(hz) / 600)


def hz_to_erb(hz):
    a = (1000 * np.log(10)) / (24.7 * 4.37)
    return a * np.log10(1 + 0.00437 * np.asarray(hz))


class BandsplitSpecification:
    def __init__(self, nfft: int, fs: int) -> None:
        self.fs = fs
        self.nfft = nfft
        self.nyquist = fs / 2
        self.max_index = nfft // 2 + 1

        self.split500 = self.hertz_to_index(500)
        self.split1k = self.hertz_to_index(1000)
        self.split2k = self.hertz_to_index(2000)
        self.split4k = self.hertz_to_index(4000)
        self.split8k = self.hertz_to_index(8000)
        self.split16k = self.hertz_to_index(16000)
        self.split20k = self.hertz_to_index(20000)

        self.above20k = [(self.split20k, self.max_index)]
        self.above16k = [(self.split16k, self.split20k)] + self.above20k

    def index_to_hertz(self, index: int):
        return index * self.fs / self.nfft

    def hertz_to_index(self, hz: float, round: bool = True):
        index = hz * self.nfft / self.fs
        if round:
            index = int(np.round(index))
        return index

    def get_band_specs_with_bandwidth(self, start_index, end_index, bandwidth_hz):
        band_specs = []
        lower = start_index
        while lower < end_index:
            upper = min(int(np.floor(lower + self.hertz_to_index(bandwidth_hz))), end_index)
            band_specs.append((lower, upper))
            lower = upper
        return band_specs

    def bands(self, *segments):
        return sum((self.get_band_specs_with_bandwidth(start, end, bandwidth) for start, end, bandwidth in segments), [])

    @abstractmethod
    def get_band_specs(self):
        raise NotImplementedError


class VocalBandsplitSpecification(BandsplitSpecification):
    def __init__(self, nfft: int, fs: int, version: str = "7") -> None:
        super().__init__(nfft=nfft, fs=fs)

        self.version = version

    def get_band_specs(self):
        return getattr(self, f"version{self.version}")()

    def version1(self):
        return self.bands((0, self.max_index, 1000))

    def version2(self):
        return self.bands((0, self.split16k, 1000), (self.split16k, self.split20k, 2000)) + self.above20k

    def version3(self):
        return self.bands((0, self.split8k, 1000), (self.split8k, self.split16k, 2000)) + self.above16k

    def version4(self):
        return (
            self.bands((0, self.split1k, 100), (self.split1k, self.split8k, 1000), (self.split8k, self.split16k, 2000))
            + self.above16k
        )

    def version5(self):
        return (
            self.bands((0, self.split1k, 100), (self.split1k, self.split16k, 1000), (self.split16k, self.split20k, 2000))
            + self.above20k
        )

    def version6(self):
        return (
            self.bands(
                (0, self.split1k, 100),
                (self.split1k, self.split4k, 500),
                (self.split4k, self.split8k, 1000),
                (self.split8k, self.split16k, 2000),
            )
            + self.above16k
        )

    def version7(self):
        return (
            self.bands(
                (0, self.split1k, 100),
                (self.split1k, self.split4k, 250),
                (self.split4k, self.split8k, 500),
                (self.split8k, self.split16k, 1000),
                (self.split16k, self.split20k, 2000),
            )
            + self.above20k
        )


class OtherBandsplitSpecification(VocalBandsplitSpecification):
    def __init__(self, nfft: int, fs: int) -> None:
        super().__init__(nfft=nfft, fs=fs, version="7")


class BassBandsplitSpecification(BandsplitSpecification):
    def __init__(self, nfft: int, fs: int, version: str = "7") -> None:
        super().__init__(nfft=nfft, fs=fs)

    def get_band_specs(self):
        return self.bands(
            (0, self.split500, 50),
            (self.split500, self.split1k, 100),
            (self.split1k, self.split4k, 500),
            (self.split4k, self.split8k, 1000),
            (self.split8k, self.split16k, 2000),
        ) + [(self.split16k, self.max_index)]


class DrumBandsplitSpecification(BandsplitSpecification):
    def __init__(self, nfft: int, fs: int) -> None:
        super().__init__(nfft=nfft, fs=fs)

    def get_band_specs(self):
        return self.bands(
            (0, self.split1k, 50),
            (self.split1k, self.split2k, 100),
            (self.split2k, self.split4k, 250),
            (self.split4k, self.split8k, 500),
            (self.split8k, self.split16k, 1000),
        ) + [(self.split16k, self.max_index)]


class PerceptualBandsplitSpecification(BandsplitSpecification):
    def __init__(
        self,
        nfft: int,
        fs: int,
        fbank_fn: Callable[[int, int, float, float, int], torch.Tensor],
        n_bands: int,
        f_min: float = 0.0,
        f_max: float = None,
    ) -> None:
        super().__init__(nfft=nfft, fs=fs)
        self.n_bands = n_bands
        if f_max is None:
            f_max = fs / 2

        self.filterbank = fbank_fn(n_bands, fs, f_min, f_max, self.max_index)

        weight_per_bin = torch.sum(self.filterbank, dim=0, keepdim=True)
        normalized_mel_fb = self.filterbank / weight_per_bin  # (n_mels, n_freqs)

        freq_weights = []
        band_specs = []
        for i in range(self.n_bands):
            active_bins = torch.nonzero(self.filterbank[i, :]).squeeze().tolist()
            if isinstance(active_bins, int):
                active_bins = (active_bins, active_bins)
            if len(active_bins) == 0:
                continue
            band_specs.append((start_index := active_bins[0], end_index := active_bins[-1] + 1))
            freq_weights.append(normalized_mel_fb[i, start_index:end_index])

        self.freq_weights = freq_weights
        self.band_specs = band_specs

    def get_band_specs(self):
        return self.band_specs

    def get_freq_weights(self):
        return self.freq_weights

    def save_to_file(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)

        import pickle

        with open(os.path.join(dir_path, "mel_bandsplit_spec.pkl"), "wb") as f:
            pickle.dump({"band_specs": self.band_specs, "freq_weights": self.freq_weights, "filterbank": self.filterbank}, f)


def mel_filterbank(n_bands, fs, f_min, f_max, n_freqs):
    nfft = 2 * (n_freqs - 1)
    fb = torch.as_tensor(
        _mel_filterbank(
            sr=fs,
            n_fft=nfft,
            n_mels=n_bands,
            fmin=f_min,
            fmax=f_max,
            htk=True,
            norm=None,
        )
    )
    fb[0, 0] = 1.0
    return fb


class MelBandsplitSpecification(PerceptualBandsplitSpecification):
    def __init__(self, nfft: int, fs: int, n_bands: int, f_min: float = 0.0, f_max: float = None) -> None:
        super().__init__(fbank_fn=mel_filterbank, nfft=nfft, fs=fs, n_bands=n_bands, f_min=f_min, f_max=f_max)


def musical_filterbank(n_bands, fs, f_min, f_max, n_freqs, scale="constant"):
    nfft, f_max, f_min = 2 * (n_freqs - 1), f_max or fs / 2, fs / (2 * (n_freqs - 1))
    df, bandwidth_mult = fs / nfft, np.power(2.0, np.log2(f_max / f_min) / n_bands)
    hz_pts = midi_to_hz(np.linspace(max(0, hz_to_midi(f_min)), hz_to_midi(f_max), n_bands))
    low_bins, high_bins = np.floor(hz_pts / bandwidth_mult / df).astype(int), np.ceil(hz_pts * bandwidth_mult / df).astype(int)

    fb = np.zeros((n_bands, n_freqs))

    for i in range(n_bands):
        fb[i, low_bins[i] : high_bins[i] + 1] = 1.0

    fb[0, : low_bins[0]] = 1.0
    fb[-1, high_bins[-1] + 1 :] = 1.0

    return torch.as_tensor(fb)


class MusicalBandsplitSpecification(PerceptualBandsplitSpecification):
    def __init__(self, nfft: int, fs: int, n_bands: int, f_min: float = 0.0, f_max: float = None) -> None:
        super().__init__(fbank_fn=musical_filterbank, nfft=nfft, fs=fs, n_bands=n_bands, f_min=f_min, f_max=f_max)


def bark_filterbank(n_bands, fs, f_min, f_max, n_freqs):
    nfft = 2 * (n_freqs - 1)
    f_max = f_max or fs / 2
    centers = np.linspace(hz_to_bark(f_min), hz_to_bark(f_max), n_bands)
    bins = np.floor((nfft + 1) * (600 * np.sinh(centers / 6) / fs)).astype(int)
    start, end = int(bins[0]), int(bins[-1])
    bark_bins = hz_to_bark(np.arange(start, end) * fs / (nfft + 1))
    fb = np.zeros((n_bands, n_freqs))

    for band, center in enumerate(centers):
        diff = bark_bins - center
        values = np.zeros_like(diff)
        lower = (-1.3 <= diff) & (diff <= -0.5)
        center_mask = (-0.5 < diff) & (diff < 0.5)
        upper = (0.5 <= diff) & (diff <= 2.5)
        values[lower] = 10 ** (2.5 * (diff[lower] + 0.5))
        values[center_mask] = 1
        values[upper] = 10 ** (-(diff[upper] - 0.5))
        fb[band, start:end] = values

    return torch.as_tensor(fb)


class BarkBandsplitSpecification(PerceptualBandsplitSpecification):
    def __init__(self, nfft: int, fs: int, n_bands: int, f_min: float = 0.0, f_max: float = None) -> None:
        super().__init__(fbank_fn=bark_filterbank, nfft=nfft, fs=fs, n_bands=n_bands, f_min=f_min, f_max=f_max)


def triangular_bark_filterbank(n_bands, fs, f_min, f_max, n_freqs):
    return triangular_filterbank_from_points(
        torch.linspace(0, fs // 2, n_freqs),
        600 * torch.sinh(torch.linspace(hz_to_bark(f_min), hz_to_bark(f_max), n_bands + 2) / 6),
    )


class TriangularBarkBandsplitSpecification(PerceptualBandsplitSpecification):
    def __init__(self, nfft: int, fs: int, n_bands: int, f_min: float = 0.0, f_max: float = None) -> None:
        super().__init__(fbank_fn=triangular_bark_filterbank, nfft=nfft, fs=fs, n_bands=n_bands, f_min=f_min, f_max=f_max)


def minibark_filterbank(n_bands, fs, f_min, f_max, n_freqs):
    fb = bark_filterbank(n_bands, fs, f_min, f_max, n_freqs)
    fb[fb < np.sqrt(0.5)] = 0.0
    return fb


class MiniBarkBandsplitSpecification(PerceptualBandsplitSpecification):
    def __init__(self, nfft: int, fs: int, n_bands: int, f_min: float = 0.0, f_max: float = None) -> None:
        super().__init__(fbank_fn=minibark_filterbank, nfft=nfft, fs=fs, n_bands=n_bands, f_min=f_min, f_max=f_max)


def erb_filterbank(
    n_bands: int,
    fs: int,
    f_min: float,
    f_max: float,
    n_freqs: int,
) -> Tensor:
    A = (1000 * np.log(10)) / (24.7 * 4.37)
    return triangular_filterbank_from_points(
        torch.linspace(0, fs // 2, n_freqs),
        (torch.pow(10, torch.linspace(hz_to_erb(f_min), hz_to_erb(f_max), n_bands + 2) / A) - 1) / 0.00437,
    )


class EquivalentRectangularBandsplitSpecification(PerceptualBandsplitSpecification):
    def __init__(self, nfft: int, fs: int, n_bands: int, f_min: float = 0.0, f_max: float = None) -> None:
        super().__init__(fbank_fn=erb_filterbank, nfft=nfft, fs=fs, n_bands=n_bands, f_min=f_min, f_max=f_max)
