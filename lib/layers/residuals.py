from collections.abc import Sequence
from typing import Protocol, cast

import torch
from torch import nn
from torch.nn import Conv1d
from torch.nn import functional as F
from torch.nn.utils.parametrize import remove_parametrizations
from torch.nn.utils.parametrizations import weight_norm

from .norms import WN
from .utils import (
    get_padding,
    call_weight_data_normal_if_Conv,
)


def remove_weight_norm(module: nn.Module, name: str = "weight") -> nn.Module:
    remove_parametrizations(module, name, leave_parametrized=True)
    return module

LRELU_SLOPE = 0.1


class RemovesWeightNorm(Protocol):
    def remove_weight_norm(self) -> None: ...


class ResBlock1(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Sequence[int] = (1, 3, 5),
    ):
        super(ResBlock1, self).__init__()

        self.convs1 = nn.ModuleList()
        for d in dilation:
            self.convs1.append(
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                ),
            )
        self.convs1.apply(call_weight_data_normal_if_Conv)

        self.convs2 = nn.ModuleList()
        for _ in dilation:
            self.convs2.append(
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            )
        self.convs2.apply(call_weight_data_normal_if_Conv)
        self.lrelu_slope = LRELU_SLOPE

    def __call__(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return super().__call__(x, x_mask=x_mask)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, self.lrelu_slope)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, self.lrelu_slope)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)

    def __prepare_scriptable__(self):
        for l in self.convs1:
            for hook in l._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    torch.nn.utils.remove_weight_norm(l)
        for l in self.convs2:
            for hook in l._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    torch.nn.utils.remove_weight_norm(l)
        return self


class ResBlock2(torch.nn.Module):
    """
    Actually this module is not used currently
    because all configs specified "resblock": "1"
    """

    def __init__(
        self,
        channels: int,
        kernel_size=3,
        dilation: Sequence[int] = (1, 3),
    ):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList()
        for d in dilation:
            self.convs.append(
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                ),
            )
        self.convs.apply(call_weight_data_normal_if_Conv)
        self.lrelu_slope = LRELU_SLOPE

    def __call__(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return super().__call__(x, x_mask=x_mask)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for c in self.convs:
            xt = F.leaky_relu(x, self.lrelu_slope)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)

    def __prepare_scriptable__(self):
        for l in self.convs:
            for hook in l._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    torch.nn.utils.remove_weight_norm(l)
        return self


class ResidualCouplingLayer(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        p_dropout: int = 0,
        gin_channels: int = 0,
        mean_only: bool = False,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super(ResidualCouplingLayer, self).__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=p_dropout,
            gin_channels=gin_channels,
        )
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        assert self.post.bias is not None
        self.post.bias.data.zero_()

    def __call__(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor | None = None,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return super().__call__(x, x_mask, g=g, reverse=reverse)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor | None = None,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet

        x1 = (x1 - m) * torch.exp(-logs) * x_mask
        x = torch.cat([x0, x1], 1)
        return x, torch.zeros([1])

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


class ResidualCouplingBlock(nn.Module):
    class Flip(nn.Module):
        """
        torch.jit.script() Compiled functions
        can't take variable number of arguments or
        use keyword-only arguments with defaults
        """

        def forward(
            self,
            x: torch.Tensor,
            x_mask: torch.Tensor,
            g: torch.Tensor | None = None,
            reverse: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            x = torch.flip(x, [1])
            if not reverse:
                logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
                return x, logdet
            else:
                return x, torch.zeros([1], device=x.device)

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
    ):
        super(ResidualCouplingBlock, self).__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(self.Flip())

    def __call__(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor | None = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        return super().__call__(x, x_mask, g=g, reverse=reverse)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor | None = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x, _ = flow.forward(x, x_mask, g=g, reverse=reverse)
        return x

    def remove_weight_norm(self):
        for i in range(self.n_flows):
            cast(RemovesWeightNorm, self.flows[i * 2]).remove_weight_norm()

    def __prepare_scriptable__(self):
        for i in range(self.n_flows):
            for hook in self.flows[i * 2]._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    torch.nn.utils.remove_weight_norm(self.flows[i * 2])
        return self
