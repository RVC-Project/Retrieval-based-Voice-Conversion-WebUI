from typing import Dict, List, Optional, Tuple, Type

import torch
from torch import nn

from ..bandit.maskestim import (
    BaseNormMLP,
    MaskEstimationModule as _MaskEstimationModule,
    MaskEstimationModuleBase,
    MaskEstimationModuleSuperBase,
    NormMLP as _NormMLP,
    OverlappingMaskEstimationModule as _OverlappingMaskEstimationModule,
)
from ..bandit.core.model.bsrnn.utils import (
    check_no_gap,
    check_no_overlap,
    check_nonzero_bandwidth,
)


class NormMLP(_NormMLP):
    def __init__(
        self,
        emb_dim: int,
        mlp_dim: int,
        bandwidth: int,
        in_channels: Optional[int],
        hidden_activation: str = "Tanh",
        hidden_activation_kwargs=None,
        complex_mask: bool = True,
    ) -> None:
        super().__init__(
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            bandwidth=bandwidth,
            in_channels=in_channels,
            hidden_activation=hidden_activation,
            hidden_activation_kwargs=hidden_activation_kwargs,
            complex_mask=complex_mask,
            use_combined=True,
            use_checkpoint=True,
        )


class OverlappingMaskEstimationModule(_OverlappingMaskEstimationModule):
    def __init__(
        self,
        in_channels: int,
        band_specs: List[Tuple[float, float]],
        freq_weights: List[torch.Tensor],
        n_freq: int,
        emb_dim: int,
        mlp_dim: int,
        cond_dim: int = 0,
        hidden_activation: str = "Tanh",
        hidden_activation_kwargs: Dict = None,
        complex_mask: bool = True,
        norm_mlp_cls: Type[nn.Module] = NormMLP,
        norm_mlp_kwargs: Dict = None,
        use_freq_weights: bool = False,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            band_specs=band_specs,
            freq_weights=freq_weights,
            n_freq=n_freq,
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            cond_dim=cond_dim,
            hidden_activation=hidden_activation,
            hidden_activation_kwargs=hidden_activation_kwargs,
            complex_mask=complex_mask,
            norm_mlp_cls=norm_mlp_cls,
            norm_mlp_kwargs=norm_mlp_kwargs,
            use_freq_weights=use_freq_weights,
            register_all_freq_weights=False,
            allow_cond=False,
            output_dtype="complex64",
            compute_all_masks=False,
        )

    def forward(self, q):
        return super().forward(q)


class MaskEstimationModule(_MaskEstimationModule):
    def __init__(
        self,
        band_specs: List[Tuple[float, float]],
        emb_dim: int,
        mlp_dim: int,
        in_channels: Optional[int],
        hidden_activation: str = "Tanh",
        hidden_activation_kwargs: Dict = None,
        complex_mask: bool = True,
        **kwargs,
    ) -> None:
        check_nonzero_bandwidth(band_specs)
        check_no_gap(band_specs)
        check_no_overlap(band_specs)
        super().__init__(
            in_channels=in_channels,
            band_specs=band_specs,
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            hidden_activation=hidden_activation,
            hidden_activation_kwargs=hidden_activation_kwargs,
            complex_mask=complex_mask,
        )


__all__ = (
    "BaseNormMLP",
    "MaskEstimationModule",
    "MaskEstimationModuleBase",
    "MaskEstimationModuleSuperBase",
    "NormMLP",
    "OverlappingMaskEstimationModule",
)
