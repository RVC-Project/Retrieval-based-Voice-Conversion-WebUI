from typing import Dict, List, Optional, Tuple, Union

import torch

from .._spectral import _SpectralComponent
from .core import MultiSourceMultiMaskBandSplitCoreRNN
from .utils import (
    BarkBandsplitSpecification,
    EquivalentRectangularBandsplitSpecification,
    MelBandsplitSpecification,
    MusicalBandsplitSpecification,
    TriangularBarkBandsplitSpecification,
    VocalBandsplitSpecification,
)

__all__ = ("MultiMaskMultiSourceBandSplitRNNSimple",)


def get_band_specs(band_specs, n_fft, fs, n_bands=None):
    if not isinstance(band_specs, str):
        return band_specs, None, False

    if band_specs in ["dnr:speech", "dnr:vox7", "musdb:vocals", "musdb:vox7"]:
        bsm = VocalBandsplitSpecification(nfft=n_fft, fs=fs).get_band_specs()
        freq_weights = None
        overlapping_band = False
    elif "tribark" in band_specs:
        assert n_bands is not None
        specs = TriangularBarkBandsplitSpecification(nfft=n_fft, fs=fs, n_bands=n_bands)
        bsm = specs.get_band_specs()
        freq_weights = specs.get_freq_weights()
        overlapping_band = True
    elif "bark" in band_specs:
        assert n_bands is not None
        specs = BarkBandsplitSpecification(nfft=n_fft, fs=fs, n_bands=n_bands)
        bsm = specs.get_band_specs()
        freq_weights = specs.get_freq_weights()
        overlapping_band = True
    elif "erb" in band_specs:
        assert n_bands is not None
        specs = EquivalentRectangularBandsplitSpecification(nfft=n_fft, fs=fs, n_bands=n_bands)
        bsm = specs.get_band_specs()
        freq_weights = specs.get_freq_weights()
        overlapping_band = True
    elif "musical" in band_specs:
        assert n_bands is not None
        specs = MusicalBandsplitSpecification(nfft=n_fft, fs=fs, n_bands=n_bands)
        bsm = specs.get_band_specs()
        freq_weights = specs.get_freq_weights()
        overlapping_band = True
    elif band_specs == "dnr:mel" or "mel" in band_specs:
        assert n_bands is not None
        specs = MelBandsplitSpecification(nfft=n_fft, fs=fs, n_bands=n_bands)
        bsm = specs.get_band_specs()
        freq_weights = specs.get_freq_weights()
        overlapping_band = True
    else:
        raise ValueError(f"Unsupported band_specs: {band_specs}")

    return bsm, freq_weights, overlapping_band


class MultiMaskMultiSourceBandSplitBaseSimple(_SpectralComponent):
    mps_model_backend = "torch"
    mps_model_compute_dtype = torch.float16

    def __init__(
        self,
        stems: List[str],
        band_specs: Union[str, List[Tuple[float, float]]],
        fs: int = 44100,
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
        n_bands: int = None,
    ) -> None:
        super().__init__(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window_fn=window_fn,
            wkwargs=wkwargs,
            power=power,
            center=center,
            normalized=normalized,
            pad_mode=pad_mode,
            onesided=onesided,
        )
        self.band_specs, self.freq_weights, self.overlapping_band = get_band_specs(
            band_specs,
            n_fft,
            fs,
            n_bands,
        )
        self.stems = stems

    def set_mps_model_backend(self, backend=None, compute_dtype=None):
        backend = (backend or "torch").lower()
        if backend not in ("torch", "mlx_full"):
            raise ValueError("mps_model_backend must be 'torch' or 'mlx_full'")
        self.mps_model_backend = backend
        if compute_dtype is None:
            return
        if isinstance(compute_dtype, str):
            compute_dtype = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }.get(compute_dtype.lower(), compute_dtype)
        if compute_dtype not in (torch.float16, torch.float32):
            raise ValueError("mps_model_compute_dtype must be 'float16' or 'float32'")
        self.mps_model_compute_dtype = compute_dtype

    def _use_mlx_full_forward(self, batch):
        return not self.training and self.mps_model_backend == "mlx_full" and batch.device.type == "mps"

    def mlx_forward_mx(self, raw_audio):
        from .....bandit_mlx import mlx_forward_bandit_mx

        return mlx_forward_bandit_mx(self, raw_audio, self.mps_model_compute_dtype)

    def forward(self, batch):
        if self._use_mlx_full_forward(batch):
            try:
                from .....bandit_mlx import mlx_forward_bandit

                return mlx_forward_bandit(self, batch, self.mps_model_compute_dtype)
            except Exception as exc:
                self._pymss_mlx_full_backend_error = repr(exc)
                self.mps_model_backend = "torch"
        with torch.no_grad():
            x = self.stft(batch)

        length = batch.shape[-1]
        output = self.bsrnn(x, cond=None)
        estimates = [self.istft(spec, length) for spec in output["spectrogram"].values()]
        return torch.stack(estimates, dim=1)


class MultiMaskMultiSourceBandSplitRNNSimple(MultiMaskMultiSourceBandSplitBaseSimple):
    def __init__(
        self,
        in_channel: int,
        stems: List[str],
        band_specs: Union[str, List[Tuple[float, float]]],
        fs: int = 44100,
        require_no_overlap: bool = False,
        require_no_gap: bool = True,
        normalize_channel_independently: bool = False,
        treat_channel_as_feature: bool = True,
        n_sqm_modules: int = 12,
        emb_dim: int = 128,
        rnn_dim: int = 256,
        cond_dim: int = 0,
        bidirectional: bool = True,
        rnn_type: str = "LSTM",
        mlp_dim: int = 512,
        hidden_activation: str = "Tanh",
        hidden_activation_kwargs: Optional[Dict] = None,
        complex_mask: bool = True,
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
        n_bands: int = None,
        use_freq_weights: bool = True,
        normalize_input: bool = False,
        mult_add_mask: bool = False,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__(
            stems=stems,
            band_specs=band_specs,
            fs=fs,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window_fn=window_fn,
            wkwargs=wkwargs,
            power=power,
            center=center,
            normalized=normalized,
            pad_mode=pad_mode,
            onesided=onesided,
            n_bands=n_bands,
        )

        self.bsrnn = MultiSourceMultiMaskBandSplitCoreRNN(
            stems=stems,
            band_specs=self.band_specs,
            in_channel=in_channel,
            require_no_overlap=require_no_overlap,
            require_no_gap=require_no_gap,
            normalize_channel_independently=normalize_channel_independently,
            treat_channel_as_feature=treat_channel_as_feature,
            n_sqm_modules=n_sqm_modules,
            emb_dim=emb_dim,
            rnn_dim=rnn_dim,
            bidirectional=bidirectional,
            rnn_type=rnn_type,
            mlp_dim=mlp_dim,
            cond_dim=cond_dim,
            hidden_activation=hidden_activation,
            hidden_activation_kwargs=hidden_activation_kwargs,
            complex_mask=complex_mask,
            overlapping_band=self.overlapping_band,
            freq_weights=self.freq_weights,
            n_freq=n_fft // 2 + 1,
            use_freq_weights=use_freq_weights,
            mult_add_mask=mult_add_mask,
        )

        self.normalize_input = normalize_input
        self.cond_dim = cond_dim

        if freeze_encoder:
            for param in self.bsrnn.band_split.parameters():
                param.requires_grad = False

            for param in self.bsrnn.tf_model.parameters():
                param.requires_grad = False
