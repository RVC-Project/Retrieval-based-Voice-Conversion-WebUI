from typing import cast

import torch
import numpy as np
from librosa.filters import mel

from .stft import STFT


class MelSpectrogram(torch.nn.Module):
    def __init__(
        self,
        is_half: bool,
        n_mel_channels: int,
        sampling_rate: int,
        win_length: int,
        hop_length: int,
        n_fft: int | None = None,
        mel_fmin: int = 0,
        mel_fmax: int | None = None,
        clamp: float = 1e-5,
        device: torch.device | str = torch.device("cpu"),
    ):
        super().__init__()
        if n_fft is None:
            n_fft = win_length
        mel_basis = mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=True,
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.clamp = clamp
        self.is_half = is_half

        self.stft = STFT(
            filter_length=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window="hann",
            use_torch_stft=True,
        ).to(device)

    @property
    def mel_basis(self) -> torch.Tensor:
        return cast(torch.Tensor, self._buffers["mel_basis"])

    def forward(
        self,
        audio: torch.Tensor,
        keyshift=0,
        speed=1,
        center=True,
    ):
        factor = 2 ** (keyshift / 12)
        win_length_new = int(np.round(self.win_length * factor))
        magnitude = self.stft(audio, keyshift, speed, center)
        if keyshift != 0:
            size = self.n_fft // 2 + 1
            resize = magnitude.size(1)
            if resize < size:
                magnitude = torch.nn.functional.pad(magnitude, (0, 0, 0, size - resize))
            magnitude = magnitude[:, :size, :] * self.win_length / win_length_new
        mel_output = torch.matmul(self.mel_basis, magnitude)
        if self.is_half:
            mel_output = mel_output.half()
        log_mel_spec = torch.log(torch.clamp(mel_output, min=self.clamp))
        return log_mel_spec
