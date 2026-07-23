from torch import nn, einsum
import torch.nn.functional as F


class Attend(nn.Module):
    def __init__(self, dropout=0.0, flash=False, scale=None):
        super().__init__()
        self.scale = scale
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.flash = flash

    def flash_attn(self, q, k, v):
        if self.scale is not None:
            q = q * (self.scale / (q.shape[-1] ** -0.5))

        # pytorch 2.2 auto attn: q, k, v, mask, dropout, softmax_scale
        return F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)

    def forward(self, q, k, v):
        scale = self.scale if self.scale is not None else (q.shape[-1] ** -0.5)

        if self.flash:
            return self.flash_attn(q, k, v)

        sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale
        out = einsum("b h i j, b h j d -> b h i d", self.attn_dropout(sim.softmax(dim=-1)), v)

        return out
