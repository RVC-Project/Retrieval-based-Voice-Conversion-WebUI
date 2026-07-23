from typing import Dict, List, Optional, Tuple, Type

import torch
from torch import nn
from torch.nn.modules import activation
from torch.utils.checkpoint import checkpoint_sequential

from .core.model.bsrnn.utils import (
    band_widths_from_specs,
    check_no_gap,
    check_no_overlap,
    check_nonzero_bandwidth,
)


def _resolve_channels(in_channels=None, in_channel=None):
    channels = in_channels if in_channels is not None else in_channel
    if channels is None:
        raise TypeError("in_channels is required")
    return channels


class BaseNormMLP(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        mlp_dim: int,
        bandwidth: int,
        in_channels: Optional[int] = None,
        in_channel: Optional[int] = None,
        hidden_activation: str = "Tanh",
        hidden_activation_kwargs=None,
        complex_mask: bool = True,
    ):
        super().__init__()
        if hidden_activation_kwargs is None:
            hidden_activation_kwargs = {}
        channels = _resolve_channels(in_channels, in_channel)
        self.hidden_activation_kwargs = hidden_activation_kwargs
        self.norm = nn.LayerNorm(emb_dim)
        self.hidden = nn.Sequential(
            nn.Linear(in_features=emb_dim, out_features=mlp_dim),
            activation.__dict__[hidden_activation](**hidden_activation_kwargs),
        )
        self.bandwidth = bandwidth
        self.in_channels = channels
        self.in_channel = channels
        self.complex_mask = complex_mask
        self.reim = 2 if complex_mask else 1
        self.glu_mult = 2


class NormMLP(BaseNormMLP):
    def __init__(
        self,
        emb_dim: int,
        mlp_dim: int,
        bandwidth: int,
        in_channels: Optional[int] = None,
        in_channel: Optional[int] = None,
        hidden_activation: str = "Tanh",
        hidden_activation_kwargs=None,
        complex_mask: bool = True,
        use_combined: bool = False,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__(
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            bandwidth=bandwidth,
            in_channels=in_channels,
            in_channel=in_channel,
            hidden_activation=hidden_activation,
            hidden_activation_kwargs=hidden_activation_kwargs,
            complex_mask=complex_mask,
        )
        self.output = nn.Sequential(
            nn.Linear(
                in_features=mlp_dim,
                out_features=self.bandwidth * self.in_channels * self.reim * 2,
            ),
            nn.GLU(dim=-1),
        )
        self.use_checkpoint = use_checkpoint
        if use_combined:
            self.combined = nn.Sequential(self.norm, self.hidden, self.output)

    def reshape_output(self, mb):
        batch, n_time, _ = mb.shape
        if self.complex_mask:
            mb = torch.view_as_complex(mb.reshape(batch, n_time, self.in_channels, self.bandwidth, self.reim).contiguous())
        else:
            mb = mb.reshape(batch, n_time, self.in_channels, self.bandwidth)

        return mb.permute(0, 2, 3, 1)

    def forward(self, qb):
        if hasattr(self, "combined"):
            if self.use_checkpoint:
                mb = checkpoint_sequential(self.combined, 2, qb, use_reentrant=False)
            else:
                mb = self.combined(qb)
        else:
            mb = self.output(self.hidden(self.norm(qb)))
        return self.reshape_output(mb)


class MultAddNormMLP(NormMLP):
    def __init__(
        self,
        emb_dim: int,
        mlp_dim: int,
        bandwidth: int,
        in_channels: Optional[int] = None,
        in_channel: Optional[int] = None,
        hidden_activation: str = "Tanh",
        hidden_activation_kwargs=None,
        complex_mask: bool = True,
    ) -> None:
        super().__init__(
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            bandwidth=bandwidth,
            in_channels=in_channels,
            in_channel=in_channel,
            hidden_activation=hidden_activation,
            hidden_activation_kwargs=hidden_activation_kwargs,
            complex_mask=complex_mask,
        )
        self.output2 = nn.Sequential(
            nn.Linear(
                in_features=mlp_dim,
                out_features=self.bandwidth * self.in_channels * self.reim * 2,
            ),
            nn.GLU(dim=-1),
        )

    def forward(self, qb):
        qb = self.hidden(self.norm(qb))
        return self.reshape_output(self.output(qb)), self.reshape_output(self.output2(qb))


class MaskEstimationModuleSuperBase(nn.Module):
    pass


class MaskEstimationModuleBase(MaskEstimationModuleSuperBase):
    def __init__(
        self,
        band_specs: List[Tuple[float, float]],
        emb_dim: int,
        mlp_dim: int,
        in_channels: Optional[int] = None,
        in_channel: Optional[int] = None,
        hidden_activation: str = "Tanh",
        hidden_activation_kwargs: Dict = None,
        complex_mask: bool = True,
        norm_mlp_cls: Type[nn.Module] = NormMLP,
        norm_mlp_kwargs: Dict = None,
    ) -> None:
        super().__init__()
        channels = _resolve_channels(in_channels, in_channel)
        self.band_widths = band_widths_from_specs(band_specs)
        self.n_bands = len(band_specs)
        hidden_activation_kwargs = hidden_activation_kwargs or {}
        norm_mlp_kwargs = norm_mlp_kwargs or {}
        self.norm_mlp = nn.ModuleList(
            [
                norm_mlp_cls(
                    bandwidth=self.band_widths[b],
                    emb_dim=emb_dim,
                    mlp_dim=mlp_dim,
                    in_channels=channels,
                    hidden_activation=hidden_activation,
                    hidden_activation_kwargs=hidden_activation_kwargs,
                    complex_mask=complex_mask,
                    **norm_mlp_kwargs,
                )
                for b in range(self.n_bands)
            ]
        )

    def compute_masks(self, q):
        return [nmlp(q[:, b, :, :]) for b, nmlp in enumerate(self.norm_mlp)]

    def compute_mask(self, q, b):
        return self.norm_mlp[b](q[:, b, :, :])


class OverlappingMaskEstimationModule(MaskEstimationModuleBase):
    def __init__(
        self,
        band_specs: List[Tuple[float, float]],
        freq_weights: List[torch.Tensor],
        n_freq: int,
        emb_dim: int,
        mlp_dim: int,
        in_channels: Optional[int] = None,
        in_channel: Optional[int] = None,
        cond_dim: int = 0,
        hidden_activation: str = "Tanh",
        hidden_activation_kwargs: Dict = None,
        complex_mask: bool = True,
        norm_mlp_cls: Type[nn.Module] = NormMLP,
        norm_mlp_kwargs: Dict = None,
        use_freq_weights: bool = True,
        register_all_freq_weights: bool = True,
        allow_cond: bool = True,
        output_dtype: str = "mask",
        compute_all_masks: bool = True,
    ) -> None:
        check_nonzero_bandwidth(band_specs)
        check_no_gap(band_specs)
        if cond_dim > 0 and not allow_cond:
            raise NotImplementedError

        channels = _resolve_channels(in_channels, in_channel)
        super().__init__(
            band_specs=band_specs,
            emb_dim=emb_dim + cond_dim,
            mlp_dim=mlp_dim,
            in_channels=channels,
            hidden_activation=hidden_activation,
            hidden_activation_kwargs=hidden_activation_kwargs,
            complex_mask=complex_mask,
            norm_mlp_cls=norm_mlp_cls,
            norm_mlp_kwargs=norm_mlp_kwargs,
        )
        self.n_freq = n_freq
        self.band_specs = band_specs
        self.in_channels = channels
        self.in_channel = channels
        self.cond_dim = cond_dim
        self.allow_cond = allow_cond
        self.output_dtype = output_dtype
        self.compute_all_masks = compute_all_masks

        should_register = freq_weights is not None and (register_all_freq_weights or use_freq_weights)
        self.use_freq_weights = bool(freq_weights is not None and use_freq_weights)
        if should_register:
            for i, fw in enumerate(freq_weights):
                self.register_buffer(f"freq_weights/{i}", fw)

    def _append_cond(self, q, cond):
        if cond is not None:
            batch, n_bands, n_time, _ = q.shape
            if cond.ndim == 2:
                cond = cond[:, None, None, :].expand(-1, n_bands, n_time, -1)
            elif cond.ndim == 3:
                assert cond.shape[1] == n_time
            else:
                raise ValueError(f"Invalid cond shape: {cond.shape}")
            return torch.cat([q, cond], dim=-1)

        if self.cond_dim <= 0:
            return q

        batch, n_bands, n_time, _ = q.shape
        cond = torch.ones(batch, n_bands, n_time, self.cond_dim, device=q.device, dtype=q.dtype)
        return torch.cat([q, cond], dim=-1)

    def forward(self, q, cond=None):
        if not self.allow_cond and cond is not None:
            raise NotImplementedError
        q = self._append_cond(q, cond)
        batch, n_bands, n_time, _ = q.shape

        mask_list = self.compute_masks(q) if self.compute_all_masks else None
        dtype = torch.complex64 if self.output_dtype == "complex64" else mask_list[0].dtype
        masks = torch.zeros(batch, self.in_channels, self.n_freq, n_time, device=q.device, dtype=dtype)

        for im in range(n_bands):
            fstart, fend = self.band_specs[im]
            mask = mask_list[im] if mask_list is not None else self.compute_mask(q, im)
            if self.use_freq_weights:
                mask = mask * self.get_buffer(f"freq_weights/{im}")[:, None]
            masks[:, :, fstart:fend, :] += mask

        return masks


class MaskEstimationModule(OverlappingMaskEstimationModule):
    def __init__(
        self,
        band_specs: List[Tuple[float, float]],
        emb_dim: int,
        mlp_dim: int,
        in_channels: Optional[int] = None,
        in_channel: Optional[int] = None,
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
            in_channel=in_channel,
            band_specs=band_specs,
            freq_weights=None,
            n_freq=0,
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            hidden_activation=hidden_activation,
            hidden_activation_kwargs=hidden_activation_kwargs,
            complex_mask=complex_mask,
        )

    def forward(self, q, cond=None):
        return torch.concat(self.compute_masks(q), dim=2)
