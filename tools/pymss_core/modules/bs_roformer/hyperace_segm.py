from typing import List

import torch
from torch import nn
import torch.nn.functional as F


def autopad(k, p=None):
    return p if p is not None else k // 2 if isinstance(k, int) else [x // 2 for x in k]


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv, self.bn, self.act = (
            nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
            nn.InstanceNorm2d(c2, affine=True, eps=1e-8),
            nn.SiLU() if act else nn.Identity(),
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DSConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, act=True):
        super().__init__()
        self.dwconv, self.pwconv, self.bn, self.act = (
            nn.Conv2d(c1, c1, k, s, autopad(k, p), groups=c1, bias=False),
            nn.Conv2d(c1, c2, 1, 1, 0, bias=False),
            nn.InstanceNorm2d(c2, affine=True, eps=1e-8),
            nn.SiLU() if act else nn.Identity(),
        )

    def forward(self, x):
        return self.act(self.bn(self.pwconv(self.dwconv(x))))


class DS_Bottleneck(nn.Module):
    def __init__(self, c1, c2, k=3, shortcut=True):
        super().__init__()
        self.dsconv1, self.dsconv2 = DSConv(c1, c1, k=3, s=1), DSConv(c1, c2, k=k, s=1)
        self.shortcut = shortcut and c1 == c2

    def forward(self, x):
        y = self.dsconv2(self.dsconv1(x))
        return x + y if self.shortcut else y


class DS_C3k(nn.Module):
    def __init__(self, c1, c2, n=1, k=3, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1, self.cv2 = Conv(c1, c_, 1, 1), Conv(c1, c_, 1, 1)
        self.cv3, self.m = Conv(2 * c_, c2, 1, 1), nn.Sequential(*[DS_Bottleneck(c_, c_, k=k, shortcut=True) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class DS_C3k2(nn.Module):
    def __init__(self, c1, c2, n=1, k=3, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1, self.m, self.cv2 = Conv(c1, c_, 1, 1), DS_C3k(c_, c_, n=n, k=k, e=1.0), Conv(c_, c2, 1, 1)

    def forward(self, x):
        return self.cv2(self.m(self.cv1(x)))


class AdaptiveHyperedgeGeneration(nn.Module):
    def __init__(self, in_channels, num_hyperedges, num_heads=8):
        super().__init__()
        self.num_hyperedges, self.num_heads, self.head_dim = num_hyperedges, num_heads, in_channels // num_heads
        self.global_proto = nn.Parameter(torch.randn(num_hyperedges, in_channels))
        self.context_mapper, self.query_proj = (
            nn.Linear(2 * in_channels, num_hyperedges * in_channels, bias=False),
            nn.Linear(in_channels, in_channels, bias=False),
        )
        self.scale = self.head_dim**-0.5

    def forward(self, x):
        b, n, c = x.shape
        pooled = x.transpose(1, 2)
        proto = self.global_proto.unsqueeze(0) + self.context_mapper(
            torch.cat(
                (
                    F.adaptive_avg_pool1d(pooled, 1).squeeze(-1),
                    F.adaptive_max_pool1d(pooled, 1).squeeze(-1),
                ),
                dim=1,
            )
        ).view(b, self.num_hyperedges, c)
        z = self.query_proj(x).view(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        proto = proto.view(b, self.num_hyperedges, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        return F.softmax(((z @ proto) * self.scale).mean(dim=1).permute(0, 2, 1), dim=-1)


class HypergraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.W_e, self.W_v = nn.Linear(in_channels, in_channels, bias=False), nn.Linear(in_channels, out_channels, bias=False)
        self.act = nn.SiLU()

    def forward(self, x, a):
        return x + self.act(self.W_v(torch.bmm(a.transpose(1, 2), self.act(self.W_e(torch.bmm(a, x))))))


class AdaptiveHypergraphComputation(nn.Module):
    def __init__(self, in_channels, out_channels, num_hyperedges=8, num_heads=8):
        super().__init__()
        self.adaptive_hyperedge_gen, self.hypergraph_conv = (
            AdaptiveHyperedgeGeneration(in_channels, num_hyperedges, num_heads),
            HypergraphConvolution(in_channels, out_channels),
        )

    def forward(self, x):
        b, _, h, w = x.shape
        x_flat = x.flatten(2).permute(0, 2, 1)
        return self.hypergraph_conv(x_flat, self.adaptive_hyperedge_gen(x_flat)).transpose(1, 2).view(b, -1, h, w)


class C3AH(nn.Module):
    def __init__(self, c1, c2, num_hyperedges=8, num_heads=8, e=0.5):
        super().__init__()
        c_ = int(c1 * e)
        self.cv1, self.cv2 = Conv(c1, c_, 1, 1), Conv(c1, c_, 1, 1)
        self.ahc, self.cv3 = AdaptiveHypergraphComputation(c_, c_, num_hyperedges, num_heads), Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        return self.cv3(torch.cat((self.ahc(self.cv2(x)), self.cv1(x)), dim=1))


class HyperACE(nn.Module):
    def __init__(self, in_channels: List[int], out_channels: int, num_hyperedges=8, num_heads=8, k=2, l=1, c_h=0.5, c_l=0.25):
        super().__init__()
        _, _, c4, _ = in_channels
        self.c_h, self.c_l = int(c4 * c_h), int(c4 * c_l)
        self.c_s = c4 - self.c_h - self.c_l
        if self.c_s <= 0:
            raise ValueError("Channel split error")

        self.fuse_conv = Conv(sum(in_channels), c4, 1, 1)
        self.high_order_branch = nn.ModuleList([C3AH(self.c_h, self.c_h, num_hyperedges, num_heads, e=1.0) for _ in range(k)])
        self.high_order_fuse = Conv(self.c_h * k, self.c_h, 1, 1)
        self.low_order_branch = nn.Sequential(*[DS_C3k(self.c_l, self.c_l, n=1, k=3, e=1.0) for _ in range(l)])
        self.final_fuse = Conv(c4, out_channels, 1, 1)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        b2, b3, b4, b5 = x
        size = b4.shape[2:]
        x_b = self.fuse_conv(
            torch.cat(
                (
                    F.interpolate(b2, size=size, mode="bilinear", align_corners=False),
                    F.interpolate(b3, size=size, mode="bilinear", align_corners=False),
                    b4,
                    F.interpolate(b5, size=size, mode="bilinear", align_corners=False),
                ),
                dim=1,
            )
        )
        x_h, x_l, x_s = torch.split(x_b, [self.c_h, self.c_l, self.c_s], dim=1)
        return self.final_fuse(
            torch.cat(
                (
                    self.high_order_fuse(torch.cat([m(x_h) for m in self.high_order_branch], dim=1)),
                    self.low_order_branch(x_l),
                    x_s,
                ),
                dim=1,
            )
        )


class GatedFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward(self, f_in, h):
        if f_in.shape[1] != h.shape[1]:
            raise ValueError(f"Channel mismatch: f_in={f_in.shape}, h={h.shape}")
        return f_in + self.gamma * h


class Backbone(nn.Module):
    def __init__(self, in_channels=256, base_channels=64, base_depth=3):
        super().__init__()
        c2, c3, c4, c5, c6 = base_channels, 256, 384, 512, 768
        self.stem = DSConv(in_channels, c2, k=3, s=(2, 1), p=1)
        self.p2 = nn.Sequential(DSConv(c2, c3, k=3, s=(2, 1), p=1), DS_C3k2(c3, c3, n=base_depth))
        self.p3 = nn.Sequential(DSConv(c3, c4, k=3, s=(2, 1), p=1), DS_C3k2(c4, c4, n=base_depth * 2))
        self.p4 = nn.Sequential(DSConv(c4, c5, k=3, s=2, p=1), DS_C3k2(c5, c5, n=base_depth * 2))
        self.p5 = nn.Sequential(DSConv(c5, c6, k=3, s=2, p=1), DS_C3k2(c6, c6, n=base_depth))
        self.out_channels = [c3, c4, c5, c6]

    def forward(self, x):
        x2 = self.p2(self.stem(x))
        x3 = self.p3(x2)
        x4 = self.p4(x3)
        return [x2, x3, x4, self.p5(x4)]


class Decoder(nn.Module):
    def __init__(self, encoder_channels: List[int], hyperace_out_c: int, decoder_channels: List[int]):
        super().__init__()
        c_p2, c_p3, c_p4, c_p5 = encoder_channels
        c_d2, c_d3, c_d4, c_d5 = decoder_channels
        self.h_to_d5, self.h_to_d4 = Conv(hyperace_out_c, c_d5, 1, 1), Conv(hyperace_out_c, c_d4, 1, 1)
        self.h_to_d3, self.h_to_d2 = Conv(hyperace_out_c, c_d3, 1, 1), Conv(hyperace_out_c, c_d2, 1, 1)
        self.fusion_d5, self.fusion_d4 = GatedFusion(c_d5), GatedFusion(c_d4)
        self.fusion_d3, self.fusion_d2 = GatedFusion(c_d3), GatedFusion(c_d2)
        self.skip_p5, self.skip_p4 = Conv(c_p5, c_d5, 1, 1), Conv(c_p4, c_d4, 1, 1)
        self.skip_p3, self.skip_p2 = Conv(c_p3, c_d3, 1, 1), Conv(c_p2, c_d2, 1, 1)
        self.up_d5, self.up_d4, self.up_d3 = DS_C3k2(c_d5, c_d4, n=1), DS_C3k2(c_d4, c_d3, n=1), DS_C3k2(c_d3, c_d2, n=1)
        self.final_d2 = DS_C3k2(c_d2, c_d2, n=1)

    def forward(self, enc_feats: List[torch.Tensor], h_ace: torch.Tensor):
        p2, p3, p4, p5 = enc_feats
        h = lambda x, conv: conv(F.interpolate(h_ace, size=x.shape[2:], mode="bilinear"))
        d5 = self.fusion_d5(self.skip_p5(p5), h(p5, self.h_to_d5))
        d4 = self.up_d5(F.interpolate(d5, size=p4.shape[2:], mode="bilinear")) + self.skip_p4(p4)
        d4 = self.fusion_d4(d4, h(d4, self.h_to_d4))
        d3 = self.up_d4(F.interpolate(d4, size=p3.shape[2:], mode="bilinear")) + self.skip_p3(p3)
        d3 = self.fusion_d3(d3, h(d3, self.h_to_d3))
        d2 = self.up_d3(F.interpolate(d3, size=p2.shape[2:], mode="bilinear")) + self.skip_p2(p2)
        d2 = self.fusion_d2(d2, h(d2, self.h_to_d2))
        return self.final_d2(d2)


class TFC_TDF(nn.Module):
    def __init__(self, in_c, c, l, f, bn=4):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(l):
            block = nn.Module()
            block.tfc1 = nn.Sequential(
                nn.InstanceNorm2d(in_c, affine=True, eps=1e-8), nn.SiLU(), nn.Conv2d(in_c, c, 3, 1, 1, bias=False)
            )
            block.tdf = nn.Sequential(
                nn.InstanceNorm2d(c, affine=True, eps=1e-8),
                nn.SiLU(),
                nn.Linear(f, f // bn, bias=False),
                nn.InstanceNorm2d(c, affine=True, eps=1e-8),
                nn.SiLU(),
                nn.Linear(f // bn, f, bias=False),
            )
            block.tfc2 = nn.Sequential(
                nn.InstanceNorm2d(c, affine=True, eps=1e-8), nn.SiLU(), nn.Conv2d(c, c, 3, 1, 1, bias=False)
            )
            block.shortcut = nn.Conv2d(in_c, c, 1, 1, 0, bias=False)
            self.blocks.append(block)
            in_c = c

    def forward(self, x):
        for block in self.blocks:
            shortcut, x = block.shortcut(x), block.tfc1(x)
            x = block.tfc2(x + block.tdf(x)) + shortcut
        return x


class FreqPixelShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, scale, f):
        super().__init__()
        self.scale, self.conv, self.out_conv = (
            scale,
            DSConv(in_channels, out_channels * scale),
            TFC_TDF(out_channels, out_channels, 2, f),
        )

    def forward(self, x):
        x = self.conv(x)
        b, c_r, h, w = x.shape
        out_c = c_r // self.scale
        return self.out_conv(
            x.view(b, out_c, self.scale, h, w).permute(0, 1, 3, 4, 2).contiguous().view(b, out_c, h, w * self.scale)
        )


class ProgressiveUpsampleHead(nn.Module):
    def __init__(self, in_channels, out_channels, target_bins=1025, in_bands=62):
        super().__init__()
        c, self.target_bins = in_channels, target_bins
        self.block1, self.block2 = (
            FreqPixelShuffle(c, c // 2, scale=2, f=in_bands * 2),
            FreqPixelShuffle(c // 2, c // 4, scale=2, f=in_bands * 4),
        )
        self.block3, self.block4 = (
            FreqPixelShuffle(c // 4, c // 8, scale=2, f=in_bands * 8),
            FreqPixelShuffle(c // 8, c // 16, scale=2, f=in_bands * 16),
        )
        self.final_conv = nn.Conv2d(c // 16, out_channels, kernel_size=3, stride=1, padding="same", bias=False)

    def forward(self, x):
        x = self.block4(self.block3(self.block2(self.block1(x))))
        return self.final_conv(
            x
            if x.shape[-1] == self.target_bins
            else F.interpolate(x, size=(x.shape[2], self.target_bins), mode="bilinear", align_corners=False)
        )


class SegmModel(nn.Module):
    def __init__(
        self,
        in_bands=62,
        in_dim=256,
        out_bins=1025,
        out_channels=4,
        base_channels=64,
        base_depth=2,
        num_hyperedges=32,
        num_heads=8,
    ):
        super().__init__()
        self.backbone = Backbone(in_channels=in_dim, base_channels=base_channels, base_depth=base_depth)
        enc_channels = self.backbone.out_channels
        _, _, c4, _ = enc_channels
        self.hyperace, self.decoder = (
            HyperACE(enc_channels, c4, num_hyperedges, num_heads, k=2, l=1),
            Decoder(enc_channels, c4, enc_channels),
        )
        self.upsample_head = ProgressiveUpsampleHead(enc_channels[0], out_channels, target_bins=out_bins, in_bands=in_bands)

    def forward(self, x):
        enc_feats = self.backbone(x)
        dec_feat = self.decoder(enc_feats, self.hyperace(enc_feats))
        return self.upsample_head(
            F.interpolate(dec_feat, size=(x.shape[2], dec_feat.shape[-1]), mode="bilinear", align_corners=False)
        )
