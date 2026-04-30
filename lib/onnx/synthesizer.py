from collections.abc import Sequence

import torch

from lib.layers.nsf import NSFGenerator
from lib.layers.synthesizers import SynthesizerTrnMsNSFsid as SynthesizerBase


class SynthesizerTrnMsNSFsid(SynthesizerBase):
    def __init__(
        self,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: int,
        resblock: str,
        resblock_kernel_sizes: Sequence[int],
        resblock_dilation_sizes: Sequence[Sequence[int]],
        upsample_rates: Sequence[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: Sequence[int],
        spk_embed_dim: int,
        gin_channels: int,
        sr: str | int | None,
        encoder_dim: int,
    ):
        super().__init__(
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
            encoder_dim,
            True,
        )
        self.speaker_map = None
        self.n_speaker = spk_embed_dim

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()
        self.enc_q.remove_weight_norm()

    def construct_spkmixmap(self):
        self.speaker_map = torch.zeros((self.n_speaker, 1, 1, self.gin_channels))
        for i in range(self.n_speaker):
            self.speaker_map[i] = self.emb_g(torch.LongTensor([[i]]))
        self.speaker_map = self.speaker_map.unsqueeze(0)

    @torch.jit.ignore
    def forward(self, phone, phone_lengths, pitch, nsff0, g, rnd, max_len=None):
        if self.speaker_map is not None:  # [N, S]  *  [S, B, 1, H]
            g = g.reshape((g.shape[0], g.shape[1], 1, 1, 1))  # [N, S, B, 1, 1]
            g = g * self.speaker_map  # [N, S, B, 1, H]
            g = torch.sum(g, dim=1)  # [N, 1, B, 1, H]
            g = g.transpose(0, -1).transpose(0, -2).squeeze(0)  # [B, H, N]
        else:
            g = g.unsqueeze(0)
            g = self.emb_g(g).transpose(1, 2)

        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z_p = (m_p + torch.exp(logs_p) * rnd) * x_mask
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        if not isinstance(self.dec, NSFGenerator):
            raise RuntimeError("ONNX synthesizer requires the NSF decoder")
        o = self.dec((z * x_mask)[:, :, :max_len], nsff0, g=g)
        return o
