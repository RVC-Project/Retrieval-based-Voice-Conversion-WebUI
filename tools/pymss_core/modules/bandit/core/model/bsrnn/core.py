from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from . import BandsplitCoreBase
from .bandsplit import BandSplitModule
from .maskestim import MaskEstimationModule, OverlappingMaskEstimationModule
from .tfmodel import SeqBandModellingModule

__all__ = ("MultiSourceMultiMaskBandSplitCoreRNN",)


class MultiMaskBandSplitCoreBase(BandsplitCoreBase):
    def forward(self, x, cond=None, compute_residual: bool = True):
        batch, in_chan, n_freq, n_time = x.shape
        x = x.reshape(-1, 1, n_freq, n_time)

        z = self.band_split(x)
        q = self.tf_model(z)

        out = {}
        for stem, mask_estimator in self.mask_estim.items():
            mask = mask_estimator(q, cond=cond)
            separated = self.mask(x, mask)
            out[stem] = separated.reshape(batch, in_chan, n_freq, n_time)

        return {"spectrogram": out}

    def instantiate_mask_estim(
        self,
        in_channel: int,
        stems: List[str],
        band_specs: List[Tuple[float, float]],
        emb_dim: int,
        mlp_dim: int,
        cond_dim: int,
        hidden_activation: str,
        hidden_activation_kwargs: Optional[Dict] = None,
        complex_mask: bool = True,
        overlapping_band: bool = False,
        freq_weights: Optional[List[torch.Tensor]] = None,
        n_freq: Optional[int] = None,
        use_freq_weights: bool = True,
        mult_add_mask: bool = False,
    ):
        if hidden_activation_kwargs is None:
            hidden_activation_kwargs = {}
        if mult_add_mask:
            raise NotImplementedError("Bandit mult_add_mask is not supported by the inference-only wrapper")

        stems = [stem for stem in stems if stem != "mne:+"]
        if overlapping_band:
            assert freq_weights is not None
            assert n_freq is not None
            self.mask_estim = nn.ModuleDict(
                {
                    stem: OverlappingMaskEstimationModule(
                        band_specs=band_specs,
                        freq_weights=freq_weights,
                        n_freq=n_freq,
                        emb_dim=emb_dim,
                        mlp_dim=mlp_dim,
                        in_channel=in_channel,
                        hidden_activation=hidden_activation,
                        hidden_activation_kwargs=hidden_activation_kwargs,
                        complex_mask=complex_mask,
                        use_freq_weights=use_freq_weights,
                    )
                    for stem in stems
                }
            )
        else:
            self.mask_estim = nn.ModuleDict(
                {
                    stem: MaskEstimationModule(
                        band_specs=band_specs,
                        emb_dim=emb_dim,
                        mlp_dim=mlp_dim,
                        cond_dim=cond_dim,
                        in_channel=in_channel,
                        hidden_activation=hidden_activation,
                        hidden_activation_kwargs=hidden_activation_kwargs,
                        complex_mask=complex_mask,
                    )
                    for stem in stems
                }
            )

    def instantiate_bandsplit(
        self,
        in_channel: int,
        band_specs: List[Tuple[float, float]],
        require_no_overlap: bool = False,
        require_no_gap: bool = True,
        normalize_channel_independently: bool = False,
        treat_channel_as_feature: bool = True,
        emb_dim: int = 128,
    ):
        self.band_split = BandSplitModule(
            in_channel=in_channel,
            band_specs=band_specs,
            require_no_overlap=require_no_overlap,
            require_no_gap=require_no_gap,
            normalize_channel_independently=normalize_channel_independently,
            treat_channel_as_feature=treat_channel_as_feature,
            emb_dim=emb_dim,
        )


class MultiSourceMultiMaskBandSplitCoreRNN(MultiMaskBandSplitCoreBase):
    def __init__(
        self,
        in_channel: int,
        stems: List[str],
        band_specs: List[Tuple[float, float]],
        require_no_overlap: bool = False,
        require_no_gap: bool = True,
        normalize_channel_independently: bool = False,
        treat_channel_as_feature: bool = True,
        n_sqm_modules: int = 12,
        emb_dim: int = 128,
        rnn_dim: int = 256,
        bidirectional: bool = True,
        rnn_type: str = "LSTM",
        mlp_dim: int = 512,
        cond_dim: int = 0,
        hidden_activation: str = "Tanh",
        hidden_activation_kwargs: Optional[Dict] = None,
        complex_mask: bool = True,
        overlapping_band: bool = False,
        freq_weights: Optional[List[torch.Tensor]] = None,
        n_freq: Optional[int] = None,
        use_freq_weights: bool = True,
        mult_add_mask: bool = False,
    ) -> None:
        super().__init__()
        self.instantiate_bandsplit(
            in_channel=in_channel,
            band_specs=band_specs,
            require_no_overlap=require_no_overlap,
            require_no_gap=require_no_gap,
            normalize_channel_independently=normalize_channel_independently,
            treat_channel_as_feature=treat_channel_as_feature,
            emb_dim=emb_dim,
        )

        self.tf_model = SeqBandModellingModule(
            n_modules=n_sqm_modules,
            emb_dim=emb_dim,
            rnn_dim=rnn_dim,
            bidirectional=bidirectional,
            rnn_type=rnn_type,
        )

        self.instantiate_mask_estim(
            in_channel=in_channel,
            stems=stems,
            band_specs=band_specs,
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            cond_dim=cond_dim,
            hidden_activation=hidden_activation,
            hidden_activation_kwargs=hidden_activation_kwargs,
            complex_mask=complex_mask,
            overlapping_band=overlapping_band,
            freq_weights=freq_weights,
            n_freq=n_freq,
            use_freq_weights=use_freq_weights,
            mult_add_mask=mult_add_mask,
        )
