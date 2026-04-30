from typing import Literal, overload, cast

import numpy as np
import torch
import torch.nn.functional as F
from librosa.util import pad_center
from scipy.signal import get_window


class STFT(torch.nn.Module):
    def __init__(
        self,
        filter_length: int = 1024,
        hop_length: int = 512,
        win_length: int | None = None,
        window: str = "hann",
        use_torch_stft: bool = True,
    ):
        """
        This module implements an STFT using 1D convolution and 1D transpose convolutions.
        This is a bit tricky so there are some cases that probably won't work as working
        out the same sizes before and after in all overlap add setups is tough. Right now,
        this code should work with hop lengths that are half the filter length (50% overlap
        between frames).

        Keyword Arguments:
            filter_length {int} -- Length of filters used (default: {1024})
            hop_length {int} -- Hop length of STFT (restrict to 50% overlap between frames) (default: {512})
            win_length {[type]} -- Length of the window function applied to each frame (if not specified, it
                equals the filter length). (default: {None})
            window {str} -- Type of window to use (options are bartlett, hann, hamming, blackman, blackmanharris)
                (default: {'hann'})
        """
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.pad_amount = int(self.filter_length / 2)
        self.win_length = filter_length if win_length is None else win_length
        self.hann_window: dict[str, torch.Tensor] = {}
        self.use_torch_stft = use_torch_stft

        if use_torch_stft:
            return

        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])]
        )
        forward_basis = torch.FloatTensor(fourier_basis)
        inverse_basis = torch.FloatTensor(np.linalg.pinv(fourier_basis))

        assert filter_length >= self.win_length

        # get window and zero center pad it to filter_length
        fft_window = get_window(window, self.win_length, fftbins=True)
        fft_window = pad_center(fft_window, size=filter_length)
        fft_window = torch.from_numpy(fft_window).float()

        # window the bases
        forward_basis *= fft_window
        inverse_basis = (inverse_basis.T * fft_window).T

        self.register_buffer("forward_basis", forward_basis.float())
        self.register_buffer("inverse_basis", inverse_basis.float())
        self.register_buffer("fft_window", fft_window.float())

    @property
    def forward_basis(self) -> torch.Tensor:
        return cast(torch.Tensor, self._buffers["forward_basis"])

    @property
    def inverse_basis(self) -> torch.Tensor:
        return cast(torch.Tensor, self._buffers["inverse_basis"])

    @property
    def fft_window(self) -> torch.Tensor:
        return cast(torch.Tensor, self._buffers["fft_window"])

    def __call__(
        self,
        input_data: torch.Tensor,
        keyshift: int = 0,
        speed: int = 1,
        center: bool = True,
    ) -> torch.Tensor:
        return super().__call__(input_data, keyshift, speed, center)

    @overload
    def transform(
        self,
        input_data: torch.Tensor,
        return_phase: Literal[False] = False,
    ) -> torch.Tensor: ...

    @overload
    def transform(
        self,
        input_data: torch.Tensor,
        return_phase: Literal[True],
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def transform(
        self,
        input_data: torch.Tensor,
        return_phase: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Take input data (audio) to STFT domain.

        Arguments:
            input_data {tensor} -- Tensor of floats, with shape (num_batch, num_samples)

        Returns:
            magnitude {tensor} -- Magnitude of STFT with shape (num_batch,
                num_frequencies, num_frames)
            phase {tensor} -- Phase of STFT with shape (num_batch,
                num_frequencies, num_frames)
        """
        input_data = F.pad(
            input_data,
            (self.pad_amount, self.pad_amount),
            mode="reflect",
        )
        forward_transform = input_data.unfold(
            1, self.filter_length, self.hop_length
        ).permute(0, 2, 1)
        forward_transform = torch.matmul(self.forward_basis, forward_transform)
        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]
        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        if return_phase:
            phase = torch.atan2(imag_part.data, real_part.data)
            return magnitude, phase
        else:
            return magnitude

    def inverse(
        self,
        magnitude: torch.Tensor,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        """Call the inverse STFT (iSTFT), given magnitude and phase tensors produced
        by the ```transform``` function.

        Arguments:
            magnitude {tensor} -- Magnitude of STFT with shape (num_batch,
                num_frequencies, num_frames)
            phase {tensor} -- Phase of STFT with shape (num_batch,
                num_frequencies, num_frames)

        Returns:
            inverse_transform {tensor} -- Reconstructed audio given magnitude and phase. Of
                shape (num_batch, num_samples)
        """
        cat = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1
        )
        fold = torch.nn.Fold(
            output_size=(1, (cat.size(-1) - 1) * self.hop_length + self.filter_length),
            kernel_size=(1, self.filter_length),
            stride=(1, self.hop_length),
        )
        inverse_transform = torch.matmul(self.inverse_basis, cat)
        inverse_transform: torch.Tensor = fold(inverse_transform)[
            :, 0, 0, self.pad_amount : -self.pad_amount
        ]
        window_square_sum = (
            self.fft_window.pow(2).repeat(cat.size(-1), 1).T.unsqueeze(0)
        )
        window_square_sum = fold(window_square_sum)[
            :, 0, 0, self.pad_amount : -self.pad_amount
        ]
        inverse_transform /= window_square_sum
        return inverse_transform

    def forward(
        self,
        input_data: torch.Tensor,
        keyshift: int = 0,
        speed: int = 1,
        center: bool = True,
    ) -> torch.Tensor:
        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(self.filter_length * factor))
        win_length_new = int(np.round(self.win_length * factor))
        hop_length_new = int(np.round(self.hop_length * speed))
        if self.use_torch_stft:
            keyshift_key = f"{keyshift}_{input_data.device}"
            if keyshift_key not in self.hann_window:
                self.hann_window[keyshift_key] = torch.hann_window(
                    self.win_length,
                ).to(input_data.device)
            fft = torch.stft(
                input_data,
                n_fft=n_fft_new,
                hop_length=hop_length_new,
                win_length=win_length_new,
                window=self.hann_window[keyshift_key],
                center=center,
                return_complex=True,
            )
            return torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))
        return self.transform(input_data)
