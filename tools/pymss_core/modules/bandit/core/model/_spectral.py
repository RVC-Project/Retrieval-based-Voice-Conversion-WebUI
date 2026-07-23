from typing import Dict, Optional

import torch
from torch import nn


class _TorchSpectrogram(nn.Module):
    def __init__(
        self,
        n_fft,
        win_length,
        hop_length,
        window_fn,
        wkwargs,
        normalized,
        center,
        pad_mode,
        onesided,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length or n_fft
        self.hop_length = hop_length
        self.normalized = normalized
        self.center = center
        self.pad_mode = pad_mode
        self.onesided = onesided
        self.register_buffer("window", window_fn(self.win_length, **(wkwargs or {})))

    def forward(self, x):
        leading_shape = x.shape[:-1]
        spec = torch.stft(
            x.reshape(-1, x.shape[-1]),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=self.normalized,
            onesided=self.onesided,
            return_complex=True,
        )
        return spec.reshape(*leading_shape, *spec.shape[-2:])


class _TorchInverseSpectrogram(_TorchSpectrogram):
    def forward(self, x, length=None):
        leading_shape = x.shape[:-2]
        audio = torch.istft(
            x.reshape(-1, *x.shape[-2:]),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            normalized=self.normalized,
            onesided=self.onesided,
            length=length,
            return_complex=False,
        )
        return audio.reshape(*leading_shape, audio.shape[-1])


class _SpectralComponent(nn.Module):
    def __init__(
        self,
        n_fft: int = 2048,
        win_length: Optional[int] = 2048,
        hop_length: int = 512,
        window_fn: str = "hann_window",
        wkwargs: Optional[Dict] = None,
        power: Optional[int] = None,
        center: bool = True,
        normalized: bool = True,
        pad_mode: str = "constant",
        onesided: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        assert power is None
        window_fn = torch.__dict__[window_fn]
        kwargs = dict(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window_fn=window_fn,
            wkwargs=wkwargs,
            normalized=normalized,
            center=center,
            pad_mode=pad_mode,
            onesided=onesided,
        )
        self.stft = _TorchSpectrogram(**kwargs)
        self.istft = _TorchInverseSpectrogram(**kwargs)
