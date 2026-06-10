import logging
from typing import Optional

logger = logging.getLogger(__name__)

import torch
from torch import nn

from infer.lib.infer_pack import commons
from infer.lib.infer_pack.discriminators import (
    DiscriminatorP,
    DiscriminatorS,
    MultiPeriodDiscriminator,
    MultiPeriodDiscriminatorV2,
)
from infer.lib.infer_pack.encoders import (
    PosteriorEncoder,
    ResidualCouplingBlock,
    TextEncoder,
)
from infer.lib.infer_pack.generators import (
    Generator,
    GeneratorNSF,
    SineGen,
    SourceModuleHnNSF,
)

__all__ = [
    "TextEncoder",
    "ResidualCouplingBlock",
    "PosteriorEncoder",
    "Generator",
    "SineGen",
    "SourceModuleHnNSF",
    "GeneratorNSF",
    "SynthesizerTrnMs256NSFsid",
    "SynthesizerTrnMs768NSFsid",
    "SynthesizerTrnMs256NSFsid_nono",
    "SynthesizerTrnMs768NSFsid_nono",
    "MultiPeriodDiscriminator",
    "MultiPeriodDiscriminatorV2",
    "DiscriminatorS",
    "DiscriminatorP",
]


sr2sr = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


class SynthesizerTrnMs256NSFsid(nn.Module):
    def __init__(
        self,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        spk_embed_dim,
        gin_channels,
        sr,
        **kwargs
    ):
        super(SynthesizerTrnMs256NSFsid, self).__init__()
        if isinstance(sr, str):
            sr = sr2sr[sr]
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        # self.hop_length = hop_length#
        self.spk_embed_dim = spk_embed_dim
        self.enc_p = TextEncoder(
            256,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            float(p_dropout),
        )
        self.dec = GeneratorNSF(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
            sr=sr,
            is_half=kwargs["is_half"],
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels
        )
        self.emb_g = nn.Embedding(self.spk_embed_dim, gin_channels)
        logger.debug(
            "gin_channels: "
            + str(gin_channels)
            + ", self.spk_embed_dim: "
            + str(self.spk_embed_dim)
        )

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()
        if hasattr(self, "enc_q"):
            self.enc_q.remove_weight_norm()

    def __prepare_scriptable__(self):
        for hook in self.dec._forward_pre_hooks.values():
            # The hook we want to remove is an instance of WeightNorm class, so
            # normally we would do `if isinstance(...)` but this class is not accessible
            # because of shadowing, so we check the module name directly.
            # https://github.com/pytorch/pytorch/blob/be0ca00c5ce260eb5bcec3237357f7a30cc08983/torch/nn/utils/__init__.py#L3
            if (
                hook.__module__ == "torch.nn.utils.weight_norm"
                and hook.__class__.__name__ == "WeightNorm"
            ):
                torch.nn.utils.remove_weight_norm(self.dec)
        for hook in self.flow._forward_pre_hooks.values():
            if (
                hook.__module__ == "torch.nn.utils.weight_norm"
                and hook.__class__.__name__ == "WeightNorm"
            ):
                torch.nn.utils.remove_weight_norm(self.flow)
        if hasattr(self, "enc_q"):
            for hook in self.enc_q._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    torch.nn.utils.remove_weight_norm(self.enc_q)
        return self

    @torch.jit.ignore
    def forward(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: torch.Tensor,
        pitchf: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        ds: Optional[torch.Tensor] = None,
    ):  # 这里ds是id，[bs,1]
        # print(1,pitch.shape)#[bs,t]
        g = self.emb_g(ds).unsqueeze(-1)  # [b, 256, 1]##1是t，广播的
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)
        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size
        )
        # print(-1,pitchf.shape,ids_slice,self.segment_size,self.hop_length,self.segment_size//self.hop_length)
        pitchf = commons.slice_segments2(pitchf, ids_slice, self.segment_size)
        # print(-2,pitchf.shape,z_slice.shape)
        o = self.dec(z_slice, pitchf, g=g)
        return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    @torch.jit.export
    def infer(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: torch.Tensor,
        nsff0: torch.Tensor,
        sid: torch.Tensor,
        skip_head: Optional[torch.Tensor] = None,
        return_length: Optional[torch.Tensor] = None,
        return_length2: Optional[torch.Tensor] = None,
    ):
        g = self.emb_g(sid).unsqueeze(-1)
        if skip_head is not None and return_length is not None:
            assert isinstance(skip_head, torch.Tensor)
            assert isinstance(return_length, torch.Tensor)
            head = int(skip_head.item())
            length = int(return_length.item())
            flow_head = torch.clamp(skip_head - 24, min=0)
            dec_head = head - int(flow_head.item())
            m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths, flow_head)
            z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
            z = self.flow(z_p, x_mask, g=g, reverse=True)
            z = z[:, :, dec_head : dec_head + length]
            x_mask = x_mask[:, :, dec_head : dec_head + length]
            nsff0 = nsff0[:, head : head + length]
        else:
            m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
            z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
            z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec(z * x_mask, nsff0, g=g, n_res=return_length2)
        return o, x_mask, (z, z_p, m_p, logs_p)


class SynthesizerTrnMs768NSFsid(SynthesizerTrnMs256NSFsid):
    def __init__(
        self,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        spk_embed_dim,
        gin_channels,
        sr,
        **kwargs
    ):
        super(SynthesizerTrnMs768NSFsid, self).__init__(
            spec_channels,
            segment_size,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            spk_embed_dim,
            gin_channels,
            sr,
            **kwargs
        )
        del self.enc_p
        self.enc_p = TextEncoder(
            768,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            float(p_dropout),
        )


class SynthesizerTrnMs256NSFsid_nono(nn.Module):
    def __init__(
        self,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        spk_embed_dim,
        gin_channels,
        sr=None,
        **kwargs
    ):
        super(SynthesizerTrnMs256NSFsid_nono, self).__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        # self.hop_length = hop_length#
        self.spk_embed_dim = spk_embed_dim
        self.enc_p = TextEncoder(
            256,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            float(p_dropout),
            f0=False,
        )
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels
        )
        self.emb_g = nn.Embedding(self.spk_embed_dim, gin_channels)
        logger.debug(
            "gin_channels: "
            + str(gin_channels)
            + ", self.spk_embed_dim: "
            + str(self.spk_embed_dim)
        )

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()
        if hasattr(self, "enc_q"):
            self.enc_q.remove_weight_norm()

    def __prepare_scriptable__(self):
        for hook in self.dec._forward_pre_hooks.values():
            # The hook we want to remove is an instance of WeightNorm class, so
            # normally we would do `if isinstance(...)` but this class is not accessible
            # because of shadowing, so we check the module name directly.
            # https://github.com/pytorch/pytorch/blob/be0ca00c5ce260eb5bcec3237357f7a30cc08983/torch/nn/utils/__init__.py#L3
            if (
                hook.__module__ == "torch.nn.utils.weight_norm"
                and hook.__class__.__name__ == "WeightNorm"
            ):
                torch.nn.utils.remove_weight_norm(self.dec)
        for hook in self.flow._forward_pre_hooks.values():
            if (
                hook.__module__ == "torch.nn.utils.weight_norm"
                and hook.__class__.__name__ == "WeightNorm"
            ):
                torch.nn.utils.remove_weight_norm(self.flow)
        if hasattr(self, "enc_q"):
            for hook in self.enc_q._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    torch.nn.utils.remove_weight_norm(self.enc_q)
        return self

    @torch.jit.ignore
    def forward(self, phone, phone_lengths, y, y_lengths, ds):  # 这里ds是id，[bs,1]
        g = self.emb_g(ds).unsqueeze(-1)  # [b, 256, 1]##1是t，广播的
        m_p, logs_p, x_mask = self.enc_p(phone, None, phone_lengths)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)
        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size
        )
        o = self.dec(z_slice, g=g)
        return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    @torch.jit.export
    def infer(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        sid: torch.Tensor,
        skip_head: Optional[torch.Tensor] = None,
        return_length: Optional[torch.Tensor] = None,
        return_length2: Optional[torch.Tensor] = None,
    ):
        g = self.emb_g(sid).unsqueeze(-1)
        if skip_head is not None and return_length is not None:
            assert isinstance(skip_head, torch.Tensor)
            assert isinstance(return_length, torch.Tensor)
            head = int(skip_head.item())
            length = int(return_length.item())
            flow_head = torch.clamp(skip_head - 24, min=0)
            dec_head = head - int(flow_head.item())
            m_p, logs_p, x_mask = self.enc_p(phone, None, phone_lengths, flow_head)
            z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
            z = self.flow(z_p, x_mask, g=g, reverse=True)
            z = z[:, :, dec_head : dec_head + length]
            x_mask = x_mask[:, :, dec_head : dec_head + length]
        else:
            m_p, logs_p, x_mask = self.enc_p(phone, None, phone_lengths)
            z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
            z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec(z * x_mask, g=g, n_res=return_length2)
        return o, x_mask, (z, z_p, m_p, logs_p)


class SynthesizerTrnMs768NSFsid_nono(SynthesizerTrnMs256NSFsid_nono):
    def __init__(
        self,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        spk_embed_dim,
        gin_channels,
        sr=None,
        **kwargs
    ):
        super(SynthesizerTrnMs768NSFsid_nono, self).__init__(
            spec_channels,
            segment_size,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            spk_embed_dim,
            gin_channels,
            sr,
            **kwargs
        )
        del self.enc_p
        self.enc_p = TextEncoder(
            768,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            float(p_dropout),
            f0=False,
        )
