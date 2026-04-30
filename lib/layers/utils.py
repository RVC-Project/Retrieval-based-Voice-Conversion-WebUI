from collections.abc import Iterator, Sequence

import torch


def call_weight_data_normal_if_Conv(m: torch.nn.Module) -> None:
    if isinstance(m, torch.nn.modules.conv._ConvNd):
        mean = 0.0
        std = 0.01
        with torch.no_grad():
            m.weight.normal_(mean, std)


def get_padding(kernel_size: int, dilation=1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def slice_on_last_dim(
    x: torch.Tensor,
    start_indices: Sequence[int] | torch.Tensor,
    segment_size: int = 4,
) -> torch.Tensor:
    new_shape = [*x.shape]
    new_shape[-1] = segment_size
    ret = torch.empty(new_shape, device=x.device)
    for i in range(x.size(0)):
        idx_str = int(start_indices[i])
        idx_end = idx_str + segment_size
        ret[i, ..., :] = x[i, ..., idx_str:idx_end]
    return ret


def rand_slice_segments_on_last_dim(
    x: torch.Tensor,
    x_lengths: int | None = None,
    segment_size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    b, _, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_on_last_dim(x, ids_str, segment_size)
    return ret, ids_str


@torch.jit.script
def activate_add_tanh_sigmoid_multiply(
    input_a: torch.Tensor, input_b: torch.Tensor, n_channels: int
) -> torch.Tensor:
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels, :])
    s_act = torch.sigmoid(in_act[:, n_channels:, :])
    acts = t_act * s_act
    return acts


def sequence_mask(
    length: torch.Tensor,
    max_length: int | None = None,
) -> torch.Tensor:
    if max_length is None:
        max_length = int(length.max())
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def total_grad_norm(
    parameters: Iterator[torch.nn.Parameter],
    norm_type: float = 2.0,
) -> float:
    norm_type = float(norm_type)
    total_norm = 0.0

    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(norm_type)
        total_norm += float(param_norm.item()) ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)

    return total_norm
