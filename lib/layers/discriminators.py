from typing import List, Tuple

import torch
from torch import nn
from torch.nn import Conv1d, Conv2d
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
from torch.nn.utils.parametrizations import weight_norm

from .residuals import LRELU_SLOPE
from .utils import get_padding


class MultiPeriodDiscriminator(torch.nn.Module):
    """
    version: 'v1' or 'v2'
    """

    def __init__(
        self, version: str, use_spectral_norm: bool = False, has_xpu: bool = False
    ):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = (
            (2, 3, 5, 7, 11, 17) if version == "v1" else (2, 3, 5, 7, 11, 17, 23, 37)
        )

        self.discriminators = nn.ModuleList(
            [
                DiscriminatorS(use_spectral_norm=use_spectral_norm),
                *(
                    DiscriminatorP(
                        i, use_spectral_norm=use_spectral_norm, has_xpu=has_xpu
                    )
                    for i in periods
                ),
            ]
        )

    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:
        return super().__call__(y, y_hat)

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm: bool = False):
        super(DiscriminatorS, self).__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorP(torch.nn.Module):
    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
        has_xpu: bool = False,
    ):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.has_xpu = has_xpu
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        sequence = (1, 32, 128, 512, 1024)
        convs_padding = (get_padding(kernel_size, 1), 0)

        self.convs = nn.ModuleList()
        for i in range(len(sequence) - 1):
            self.convs.append(
                norm_f(
                    Conv2d(
                        sequence[i],
                        sequence[i + 1],
                        (kernel_size, 1),
                        (stride, 1),
                        padding=convs_padding,
                    )
                )
            )
        self.convs.append(
            norm_f(
                Conv2d(
                    1024,
                    1024,
                    (kernel_size, 1),
                    1,
                    padding=convs_padding,
                )
            )
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            if self.has_xpu and x.dtype == torch.bfloat16:
                x = F.pad(x.to(dtype=torch.float16), (0, n_pad), "reflect").to(
                    dtype=torch.bfloat16
                )
            else:
                x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap
