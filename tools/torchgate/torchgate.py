import torch
from torch.nn.functional import conv1d, conv2d
from typing import Union, Optional
from .utils import linspace, temperature_sigmoid, amp_to_db


class TorchGate(torch.nn.Module):
    """
    A PyTorch module that applies a spectral gate to an input signal.

    Arguments:
        sr {int} -- Sample rate of the input signal.
        nonstationary {bool} -- Whether to use non-stationary or stationary masking (default: {False}).
        n_std_thresh_stationary {float} -- Number of standard deviations above mean to threshold noise for
                                           stationary masking (default: {1.5}).
        n_thresh_nonstationary {float} -- Number of multiplies above smoothed magnitude spectrogram. for
                                        non-stationary masking (default: {1.3}).
        temp_coeff_nonstationary {float} -- Temperature coefficient for non-stationary masking (default: {0.1}).
        n_movemean_nonstationary {int} -- Number of samples for moving average smoothing in non-stationary masking
                                          (default: {20}).
        prop_decrease {float} -- Proportion to decrease signal by where the mask is zero (default: {1.0}).
        n_fft {int} -- Size of FFT for STFT (default: {1024}).
        win_length {[int]} -- Window length for STFT. If None, defaults to `n_fft` (default: {None}).
        hop_length {[int]} -- Hop length for STFT. If None, defaults to `win_length` // 4 (default: {None}).
        freq_mask_smooth_hz {float} -- Frequency smoothing width for mask (in Hz). If None, no smoothing is applied
                                     (default: {500}).
        time_mask_smooth_ms {float} -- Time smoothing width for mask (in ms). If None, no smoothing is applied
                                     (default: {50}).
    """

    @torch.no_grad()
    def __init__(
        self,
        sr: int,
        nonstationary: bool = False,
        n_std_thresh_stationary: float = 1.5,
        n_thresh_nonstationary: float = 1.3,
        temp_coeff_nonstationary: float = 0.1,
        n_movemean_nonstationary: int = 20,
        prop_decrease: float = 1.0,
        n_fft: int = 1024,
        win_length: bool = None,
        hop_length: int = None,
        freq_mask_smooth_hz: float = 500,
        time_mask_smooth_ms: float = 50,
    ):
        super().__init__()

        # General Params
        self.sr = sr
        self.nonstationary = nonstationary
        assert 0.0 <= prop_decrease <= 1.0
        self.prop_decrease = prop_decrease

        # STFT Params
        self.n_fft = n_fft
        self.win_length = self.n_fft if win_length is None else win_length
        self.hop_length = self.win_length // 4 if hop_length is None else hop_length

        # Stationary Params
        self.n_std_thresh_stationary = n_std_thresh_stationary

        # Non-Stationary Params
        self.temp_coeff_nonstationary = temp_coeff_nonstationary
        self.n_movemean_nonstationary = n_movemean_nonstationary
        self.n_thresh_nonstationary = n_thresh_nonstationary

        # Smooth Mask Params
        self.freq_mask_smooth_hz = freq_mask_smooth_hz
        self.time_mask_smooth_ms = time_mask_smooth_ms
        self.register_buffer("smoothing_filter", self._generate_mask_smoothing_filter())

    @torch.no_grad()
    def _generate_mask_smoothing_filter(self) -> Union[torch.Tensor, None]:
        """
        A PyTorch module that applies a spectral gate to an input signal using the STFT.

        Returns:
            smoothing_filter (torch.Tensor): a 2D tensor representing the smoothing filter,
            with shape (n_grad_freq, n_grad_time), where n_grad_freq is the number of frequency
            bins to smooth and n_grad_time is the number of time frames to smooth.
            If both self.freq_mask_smooth_hz and self.time_mask_smooth_ms are None, returns None.
        """
        if self.freq_mask_smooth_hz is None and self.time_mask_smooth_ms is None:
            return None

        n_grad_freq = (
            1
            if self.freq_mask_smooth_hz is None
            else int(self.freq_mask_smooth_hz / (self.sr / (self.n_fft / 2)))
        )
        if n_grad_freq < 1:
            raise ValueError(
                f"freq_mask_smooth_hz needs to be at least {int((self.sr / (self._n_fft / 2)))} Hz"
            )

        n_grad_time = (
            1
            if self.time_mask_smooth_ms is None
            else int(self.time_mask_smooth_ms / ((self.hop_length / self.sr) * 1000))
        )
        if n_grad_time < 1:
            raise ValueError(
                f"time_mask_smooth_ms needs to be at least {int((self.hop_length / self.sr) * 1000)} ms"
            )

        if n_grad_time == 1 and n_grad_freq == 1:
            return None

        v_f = torch.cat(
            [
                linspace(0, 1, n_grad_freq + 1, endpoint=False),
                linspace(1, 0, n_grad_freq + 2),
            ]
        )[1:-1]
        v_t = torch.cat(
            [
                linspace(0, 1, n_grad_time + 1, endpoint=False),
                linspace(1, 0, n_grad_time + 2),
            ]
        )[1:-1]
        smoothing_filter = torch.outer(v_f, v_t).unsqueeze(0).unsqueeze(0)

        return smoothing_filter / smoothing_filter.sum()

    @torch.no_grad()
    def _stationary_mask(
        self, X_db: torch.Tensor, xn: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes a stationary binary mask to filter out noise in a log-magnitude spectrogram.

        Arguments:
            X_db (torch.Tensor): 2D tensor of shape (frames, freq_bins) containing the log-magnitude spectrogram.
            xn (torch.Tensor): 1D tensor containing the audio signal corresponding to X_db.

        Returns:
            sig_mask (torch.Tensor): Binary mask of the same shape as X_db, where values greater than the threshold
            are set to 1, and the rest are set to 0.
        """
        if xn is not None:
            XN = torch.stft(
                xn,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                return_complex=True,
                pad_mode="constant",
                center=True,
                window=torch.hann_window(self.win_length).to(xn.device),
            )

            XN_db = amp_to_db(XN).to(dtype=X_db.dtype)
        else:
            XN_db = X_db

        # calculate mean and standard deviation along the frequency axis
        std_freq_noise, mean_freq_noise = torch.std_mean(XN_db, dim=-1)

        # compute noise threshold
        noise_thresh = mean_freq_noise + std_freq_noise * self.n_std_thresh_stationary

        # create binary mask by thresholding the spectrogram
        sig_mask = X_db > noise_thresh.unsqueeze(2)
        return sig_mask

    @torch.no_grad()
    def _nonstationary_mask(self, X_abs: torch.Tensor) -> torch.Tensor:
        """
        Computes a non-stationary binary mask to filter out noise in a log-magnitude spectrogram.

        Arguments:
            X_abs (torch.Tensor): 2D tensor of shape (frames, freq_bins) containing the magnitude spectrogram.

        Returns:
            sig_mask (torch.Tensor): Binary mask of the same shape as X_abs, where values greater than the threshold
            are set to 1, and the rest are set to 0.
        """
        X_smoothed = (
            conv1d(
                X_abs.reshape(-1, 1, X_abs.shape[-1]),
                torch.ones(
                    self.n_movemean_nonstationary,
                    dtype=X_abs.dtype,
                    device=X_abs.device,
                ).view(1, 1, -1),
                padding="same",
            ).view(X_abs.shape)
            / self.n_movemean_nonstationary
        )

        # Compute slowness ratio and apply temperature sigmoid
        slowness_ratio = (X_abs - X_smoothed) / (X_smoothed + 1e-6)
        sig_mask = temperature_sigmoid(
            slowness_ratio, self.n_thresh_nonstationary, self.temp_coeff_nonstationary
        )

        return sig_mask

    def forward(
        self, x: torch.Tensor, xn: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply the proposed algorithm to the input signal.

        Arguments:
            x (torch.Tensor): The input audio signal, with shape (batch_size, signal_length).
            xn (Optional[torch.Tensor]): The noise signal used for stationary noise reduction. If `None`, the input
                                         signal is used as the noise signal. Default: `None`.

        Returns:
            torch.Tensor: The denoised audio signal, with the same shape as the input signal.
        """
        assert x.ndim == 2
        if x.shape[-1] < self.win_length * 2:
            raise Exception(f"x must be bigger than {self.win_length * 2}")

        assert xn is None or xn.ndim == 1 or xn.ndim == 2
        if xn is not None and xn.shape[-1] < self.win_length * 2:
            raise Exception(f"xn must be bigger than {self.win_length * 2}")

        # Compute short-time Fourier transform (STFT)
        X = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=True,
            pad_mode="constant",
            center=True,
            window=torch.hann_window(self.win_length).to(x.device),
        )

        # Compute signal mask based on stationary or nonstationary assumptions
        if self.nonstationary:
            sig_mask = self._nonstationary_mask(X.abs())
        else:
            sig_mask = self._stationary_mask(amp_to_db(X), xn)

        # Propagate decrease in signal power
        sig_mask = self.prop_decrease * (sig_mask * 1.0 - 1.0) + 1.0

        # Smooth signal mask with 2D convolution
        if self.smoothing_filter is not None:
            sig_mask = conv2d(
                sig_mask.unsqueeze(1),
                self.smoothing_filter.to(sig_mask.dtype),
                padding="same",
            )

        # Apply signal mask to STFT magnitude and phase components
        Y = X * sig_mask.squeeze(1)

        # Inverse STFT to obtain time-domain signal
        y = torch.istft(
            Y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=True,
            window=torch.hann_window(self.win_length).to(Y.device),
        )

        return y.to(dtype=x.dtype)
