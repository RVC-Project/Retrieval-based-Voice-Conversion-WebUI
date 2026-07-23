from typing import List, Tuple

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential

from .core.model.bsrnn.utils import (
    band_widths_from_specs,
    check_no_gap,
    check_no_overlap,
    check_nonzero_bandwidth,
)


class NormFC(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        bandwidth: int,
        in_channels: int,
        normalize_channel_independently: bool = False,
        treat_channel_as_feature: bool = True,
    ) -> None:
        super().__init__()
        self.treat_channel_as_feature = treat_channel_as_feature
        if normalize_channel_independently:
            raise NotImplementedError

        reim = 2
        self.norm = nn.LayerNorm(in_channels * bandwidth * reim)

        fc_in = bandwidth * reim
        if treat_channel_as_feature:
            fc_in *= in_channels
        else:
            assert emb_dim % in_channels == 0
            emb_dim = emb_dim // in_channels

        self.fc = nn.Linear(fc_in, emb_dim)

    def forward(self, xb):
        batch, n_time, in_channels, ribw = xb.shape
        xb = self.norm(xb.reshape(batch, n_time, in_channels * ribw))
        if self.treat_channel_as_feature:
            return self.fc(xb)
        return self.fc(xb.reshape(batch, n_time, in_channels, ribw)).reshape(batch, n_time, -1)


class SequentialNormFC(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        bandwidth: int,
        in_channels: int,
        normalize_channel_independently: bool = False,
        treat_channel_as_feature: bool = True,
    ) -> None:
        super().__init__()
        if not treat_channel_as_feature:
            raise NotImplementedError
        if normalize_channel_independently:
            raise NotImplementedError

        self.combined = nn.Sequential(
            nn.LayerNorm(in_channels * bandwidth * 2),
            nn.Linear(in_channels * bandwidth * 2, emb_dim),
        )

    def forward(self, xb):
        return checkpoint_sequential(self.combined, 1, xb, use_reentrant=False)


class BandSplitModuleBase(nn.Module):
    def __init__(
        self,
        band_specs: List[Tuple[float, float]],
        emb_dim: int,
        in_channels: int,
        norm_fc_cls: type[nn.Module],
        complex_order: str,
        flatten_input: bool,
        require_no_overlap: bool = False,
        require_no_gap: bool = True,
        normalize_channel_independently: bool = False,
        treat_channel_as_feature: bool = True,
    ) -> None:
        super().__init__()
        check_nonzero_bandwidth(band_specs)
        if require_no_gap:
            check_no_gap(band_specs)
        if require_no_overlap:
            check_no_overlap(band_specs)

        self.band_specs = band_specs
        self.band_widths = band_widths_from_specs(band_specs)
        self.n_bands = len(band_specs)
        self.emb_dim = emb_dim
        self.complex_order = complex_order
        self.flatten_input = flatten_input
        self.norm_fc_modules = nn.ModuleList(
            [
                norm_fc_cls(
                    emb_dim=emb_dim,
                    bandwidth=bw,
                    in_channels=in_channels,
                    normalize_channel_independently=normalize_channel_independently,
                    treat_channel_as_feature=treat_channel_as_feature,
                )
                for bw in self.band_widths
            ]
        )

    def _band_view(self, x):
        xr = torch.view_as_real(x)
        if self.complex_order == "reim_freq":
            return xr.permute(0, 3, 1, 4, 2)
        if self.complex_order == "freq_reim":
            return xr.permute(0, 3, 1, 2, 4).contiguous()
        raise ValueError(f"unsupported complex_order: {self.complex_order}")

    def forward(self, x: torch.Tensor):
        batch, in_channels, _, n_time = x.shape
        z = torch.zeros(
            size=(batch, self.n_bands, n_time, self.emb_dim),
            device=x.device,
        )
        xr = self._band_view(x)

        for i, nfm in enumerate(self.norm_fc_modules):
            fstart, fend = self.band_specs[i]
            if self.complex_order == "reim_freq":
                xb = xr[..., fstart:fend].reshape(batch, n_time, in_channels, -1)
            else:
                xb = xr[:, :, :, fstart:fend].reshape(batch, n_time, -1)
            z[:, i, :, :] = nfm((xb.reshape(batch, n_time, -1) if self.flatten_input else xb).contiguous())

        return z


class _ConfiguredBandSplitModule(BandSplitModuleBase):
    norm_fc_cls: type[nn.Module]
    complex_order: str
    flatten_input: bool

    def __init__(
        self,
        band_specs: List[Tuple[float, float]],
        emb_dim: int,
        in_channels: int,
        require_no_overlap: bool = False,
        require_no_gap: bool = True,
        normalize_channel_independently: bool = False,
        treat_channel_as_feature: bool = True,
    ) -> None:
        super().__init__(
            band_specs=band_specs,
            emb_dim=emb_dim,
            in_channels=in_channels,
            norm_fc_cls=self.norm_fc_cls,
            complex_order=self.complex_order,
            flatten_input=self.flatten_input,
            require_no_overlap=require_no_overlap,
            require_no_gap=require_no_gap,
            normalize_channel_independently=normalize_channel_independently,
            treat_channel_as_feature=treat_channel_as_feature,
        )
