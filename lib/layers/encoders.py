import math

import torch
from torch import nn

from .attentions import MultiHeadAttention, FFN
from .norms import LayerNorm, WN
from .utils import sequence_mask


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        window_size: int = 10,
    ):
        super(Encoder, self).__init__()

        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()

        for _ in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def __call__(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        return super().__call__(x, x_mask)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for attn, norm1, ffn, norm2 in zip(
            self.attn_layers,
            self.norm_layers_1,
            self.ffn_layers,
            self.norm_layers_2,
        ):
            y = attn(x, x, attn_mask)
            y = self.drop(y)
            x = norm1(x + y)

            y = ffn(x, x_mask)
            y = self.drop(y)
            x = norm2(x + y)
        x = x * x_mask
        return x


class TextEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        f0: bool = True,
    ):
        super(TextEncoder, self).__init__()

        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)

        self.emb_phone = nn.Linear(in_channels, hidden_channels)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        if f0 == True:
            self.emb_pitch = nn.Embedding(256, hidden_channels)  # pitch 256
        self.encoder = Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            float(p_dropout),
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def __call__(
        self,
        phone: torch.Tensor,
        pitch: torch.Tensor | None,
        lengths: torch.Tensor,
        skip_head: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return super().__call__(
            phone,
            pitch,
            lengths,
            skip_head=skip_head,
        )

    def forward(
        self,
        phone: torch.Tensor,
        pitch: torch.Tensor | None,
        lengths: torch.Tensor,
        skip_head: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.emb_phone(phone)
        if pitch is not None:
            x += self.emb_pitch(pitch)
        x = x * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = self.lrelu(x)
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(
            sequence_mask(lengths, x.size(2)),
            1,
        ).to(x.dtype)
        x = self.encoder(x * x_mask, x_mask)
        if skip_head is not None:
            head = int(skip_head)
            x = x[:, :, head:]
            x_mask = x_mask[:, :, head:]
        stats: torch.Tensor = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return m, logs, x_mask


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels=0,
    ):
        super(PosteriorEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def __call__(
        self, x: torch.Tensor, x_lengths: torch.Tensor, g: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return super().__call__(x, x_lengths, g=g)

    def forward(
        self, x: torch.Tensor, x_lengths: torch.Tensor, g: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_mask = torch.unsqueeze(
            sequence_mask(x_lengths, x.size(2)),
            1,
        ).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()

    def __prepare_scriptable__(self):
        for hook in self.enc._forward_pre_hooks.values():
            if (
                hook.__module__ == "torch.nn.utils.weight_norm"
                and hook.__class__.__name__ == "WeightNorm"
            ):
                torch.nn.utils.remove_weight_norm(self.enc)
        return self
