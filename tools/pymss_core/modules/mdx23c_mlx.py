import numpy as np
import torch

from .mlx_utils import mlx_periodic_hann_window
from .bs_roformer.mlx_attention import (
    _gelu,
    _linear,
    _mlx_dtype,
    _torch_to_mlx_array,
    mlx_to_torch_mps,
)
from .mdx23c_tfc_tdf_v3 import Downscale, TFC_TDF, Upscale


def torch_to_mlx_input(tensor, dtype):
    import mlx.core as mx

    return mx.array(tensor.detach().to(dtype=dtype).cpu().numpy())


def _mlx_param(module, name, tensor, dtype):
    cache = getattr(module, "_pymss_mlx_full_param_cache", None)
    if cache is None:
        cache = {}
        module._pymss_mlx_full_param_cache = cache
    key = (name, tensor.data_ptr(), tensor._version, tuple(tensor.shape), dtype)
    cached = cache.get(name)
    if cached is not None and cached[0] == key:
        return cached[1]
    value = _torch_to_mlx_array(tensor, dtype)
    cache[name] = (key, value)
    return value


def _hann_window(length, dtype):
    return mlx_periodic_hann_window(length, dtype)


def _reflect_pad_last(x, pad):
    import mlx.core as mx

    if pad <= 0:
        return x
    if x.shape[-1] <= pad:
        raise ValueError("reflect padding requires input length greater than padding")
    return mx.concatenate((x[..., 1 : pad + 1][..., ::-1], x, x[..., -pad - 1 : -1][..., ::-1]), axis=-1)


def _subband_stft(module, raw_audio, dtype):
    import mlx.core as mx

    if raw_audio.ndim == 2:
        raw_audio = raw_audio[None]
    batch_dims = raw_audio.shape[:-2]
    channels, audio_length = raw_audio.shape[-2:]
    n_fft = int(module.stft.n_fft)
    hop = int(module.stft.hop_length)
    dim_f = int(module.stft.dim_f)

    x = raw_audio.reshape(-1, audio_length).astype(dtype)
    x = _reflect_pad_last(x, n_fft // 2)
    frames = 1 + (x.shape[-1] - n_fft) // hop
    framed = mx.as_strided(x, shape=(x.shape[0], frames, n_fft), strides=(x.shape[-1], hop, 1))
    window = _hann_window(n_fft, dtype)
    spec = mx.fft.rfft(framed * window, n=n_fft, axis=-1)
    spec = mx.moveaxis(spec, -1, -2)
    ri = mx.stack((spec.real, spec.imag), axis=1)
    ri = ri.reshape(*batch_dims, channels * 2, ri.shape[-2], ri.shape[-1])
    return ri[..., :dim_f, :], {
        "batch_dims": batch_dims,
        "channels": channels,
        "audio_length": audio_length,
        "n_fft": n_fft,
        "hop": hop,
        "dim_f": dim_f,
        "window": window,
        "dtype": dtype,
    }


def _subband_istft(module, x, context):
    import mlx.core as mx

    batch_dims = x.shape[:-3]
    channels, freq_bins, time_bins = x.shape[-3:]
    n_fft = context["n_fft"]
    hop = context["hop"]
    full_freq_bins = n_fft // 2 + 1
    if freq_bins < full_freq_bins:
        x = mx.pad(x, [(0, 0)] * (x.ndim - 2) + [(0, full_freq_bins - freq_bins), (0, 0)])
    x = x.reshape(-1, 2, full_freq_bins, time_bins).transpose(0, 2, 3, 1)
    spec = x[..., 0] + (1j * x[..., 1])
    spec = mx.moveaxis(spec, -2, -1)
    frames = mx.fft.irfft(spec, n=n_fft, axis=-1).astype(context["dtype"]) * context["window"]

    frame_count = frames.shape[1]
    full_length = n_fft + hop * (frame_count - 1)
    positions = mx.arange(n_fft)[None, :] + hop * mx.arange(frame_count)[:, None]
    audio = mx.zeros((frames.shape[0], full_length), dtype=context["dtype"]).at[:, positions].add(frames)
    denom_frames = mx.broadcast_to(mx.square(context["window"])[None, :], (frame_count, n_fft))
    denom = mx.zeros((full_length,), dtype=context["dtype"]).at[positions].add(denom_frames)
    audio = audio / mx.maximum(denom[None, :], mx.array(1e-11, dtype=context["dtype"]))
    pad = n_fft // 2
    audio = audio[..., pad : pad + context["audio_length"]]
    return audio.reshape(*batch_dims, 2, audio.shape[-1])


def _cac_to_cws_mx(x, num_subbands):
    batch, channels, freq_bins, time_bins = x.shape
    return x.reshape(batch, channels * num_subbands, freq_bins // num_subbands, time_bins)


def _cws_to_cac_mx(x, num_subbands):
    batch, channels, freq_bins, time_bins = x.shape
    return x.reshape(batch, channels // num_subbands, freq_bins * num_subbands, time_bins)


def _conv_padding(conv):
    padding = conv.padding
    if isinstance(padding, str):
        kernel = conv.kernel_size
        return kernel[0] // 2, kernel[1] // 2
    if isinstance(padding, tuple):
        return padding
    return padding, padding


def _conv2d_nchw(conv, x, dtype):
    import mlx.core as mx

    weight = mx.transpose(_mlx_param(conv, "weight", conv.weight, dtype), (0, 2, 3, 1))
    y = mx.conv2d(
        mx.transpose(x, (0, 2, 3, 1)),
        weight,
        stride=conv.stride,
        padding=_conv_padding(conv),
        dilation=conv.dilation,
        groups=conv.groups,
    )
    if conv.bias is not None:
        y = y + _mlx_param(conv, "bias", conv.bias, dtype)
    return mx.transpose(y, (0, 3, 1, 2))


def _conv_transpose2d_nchw(conv, x, dtype):
    import mlx.core as mx

    if conv.groups != 1:
        raise TypeError("MLX MDX23C ConvTranspose2d currently supports groups=1 only")
    weight = mx.transpose(_mlx_param(conv, "weight", conv.weight, dtype), (1, 2, 3, 0))
    y = mx.conv_transpose2d(
        mx.transpose(x, (0, 2, 3, 1)),
        weight,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
    )
    if conv.bias is not None:
        y = y + _mlx_param(conv, "bias", conv.bias, dtype)
    return mx.transpose(y, (0, 3, 1, 2))


def _instance_norm2d(module, x, dtype):
    import mlx.core as mx

    x32 = x.astype(mx.float32)
    mean = mx.mean(x32, axis=(2, 3), keepdims=True)
    var = mx.mean(mx.square(x32 - mean), axis=(2, 3), keepdims=True)
    y = ((x32 - mean) * mx.rsqrt(var + module.eps)).astype(x.dtype)
    if module.affine:
        weight = _mlx_param(module, "weight", module.weight, dtype).reshape(1, -1, 1, 1)
        bias = _mlx_param(module, "bias", module.bias, dtype).reshape(1, -1, 1, 1)
        y = y * weight + bias
    return y


def _batch_norm2d(module, x, dtype):
    import mlx.core as mx

    if module.training:
        raise TypeError("MLX MDX23C BatchNorm2d supports eval mode only")
    y = x.astype(mx.float32)
    mean = _torch_to_mlx_array(module.running_mean, torch.float32).reshape(1, -1, 1, 1)
    var = _torch_to_mlx_array(module.running_var, torch.float32).reshape(1, -1, 1, 1)
    y = (y - mean) * mx.rsqrt(var + module.eps)
    if module.affine:
        weight = _mlx_param(module, "weight", module.weight, dtype).reshape(1, -1, 1, 1)
        bias = _mlx_param(module, "bias", module.bias, dtype).reshape(1, -1, 1, 1)
        y = y.astype(x.dtype) * weight + bias
    return y.astype(x.dtype)


def _group_norm(module, x, dtype):
    import mlx.core as mx

    b, c, h, w = x.shape
    groups = int(module.num_groups)
    y = x.astype(mx.float32).reshape(b, groups, c // groups, h, w)
    mean = mx.mean(y, axis=(2, 3, 4), keepdims=True)
    var = mx.mean(mx.square(y - mean), axis=(2, 3, 4), keepdims=True)
    y = ((y - mean) * mx.rsqrt(var + module.eps)).reshape(b, c, h, w).astype(x.dtype)
    if module.affine:
        weight = _mlx_param(module, "weight", module.weight, dtype).reshape(1, -1, 1, 1)
        bias = _mlx_param(module, "bias", module.bias, dtype).reshape(1, -1, 1, 1)
        y = y * weight + bias
    return y


def _activation(module, x):
    import mlx.core as mx

    if isinstance(module, torch.nn.GELU):
        if module.approximate != "none":
            raise TypeError("MLX MDX23C only supports exact GELU")
        return _gelu(x)
    if isinstance(module, torch.nn.ReLU):
        return mx.maximum(x, 0)
    if isinstance(module, torch.nn.ELU):
        return mx.where(x > 0, x, module.alpha * (mx.exp(x) - 1))
    if isinstance(module, torch.nn.Identity):
        return x
    raise TypeError(f"unsupported MDX23C activation for MLX full backend: {type(module).__name__}")


def _module_forward(module, x, dtype):
    if isinstance(module, TFC_TDF):
        return _tfc_tdf(module, x, dtype)
    if isinstance(module, (Downscale, Upscale)):
        return _module_forward(module.conv, x, dtype)
    if isinstance(module, torch.nn.Sequential):
        for child in module:
            x = _module_forward(child, x, dtype)
        return x
    if isinstance(module, torch.nn.Conv2d):
        return _conv2d_nchw(module, x, dtype)
    if isinstance(module, torch.nn.ConvTranspose2d):
        return _conv_transpose2d_nchw(module, x, dtype)
    if isinstance(module, torch.nn.Linear):
        return _linear(
            x,
            _mlx_param(module, "weight", module.weight, dtype),
            None if module.bias is None else _mlx_param(module, "bias", module.bias, dtype),
        )
    if isinstance(module, torch.nn.InstanceNorm2d):
        return _instance_norm2d(module, x, dtype)
    if isinstance(module, torch.nn.BatchNorm2d):
        return _batch_norm2d(module, x, dtype)
    if isinstance(module, torch.nn.GroupNorm):
        return _group_norm(module, x, dtype)
    if isinstance(module, (torch.nn.GELU, torch.nn.ReLU, torch.nn.ELU, torch.nn.Identity)):
        return _activation(module, x)
    raise TypeError(f"unsupported MDX23C layer for MLX full backend: {type(module).__name__}")


def _tfc_tdf(module, x, dtype):
    for block in module.blocks:
        shortcut = _conv2d_nchw(block.shortcut, x, dtype)
        x = _module_forward(block.tfc1, x, dtype)
        x = _module_forward(block.tfc2, x + _module_forward(block.tdf, x, dtype), dtype) + shortcut
    return x


def _forward_core(module, x, dtype):
    import mlx.core as mx

    encoder_outputs = []
    for block in module.encoder_blocks:
        x = _tfc_tdf(block.tfc_tdf, x, dtype)
        encoder_outputs.append(x)
        x = _module_forward(block.downscale, x, dtype)

    x = _tfc_tdf(module.bottleneck_block, x, dtype)

    for block in module.decoder_blocks:
        x = _module_forward(block.upscale, x, dtype)
        x = _tfc_tdf(block.tfc_tdf, mx.concatenate((x, encoder_outputs.pop()), axis=1), dtype)

    return x


def mlx_forward_mdx23c_mx(module, raw_audio, dtype=torch.float16):
    import mlx.core as mx

    if dtype not in (torch.float16, torch.float32):
        raise TypeError("MLX full MDX23C supports torch.float16 or torch.float32 compute dtype")
    mx_dtype = _mlx_dtype(dtype)

    x, context = _subband_stft(module, raw_audio.astype(mx_dtype), mx_dtype)
    mix = x = _cac_to_cws_mx(x, module.num_subbands)
    first_conv_out = x = _conv2d_nchw(module.first_conv, x, dtype)
    x = _forward_core(module, x.transpose(0, 1, 3, 2), dtype).transpose(0, 1, 3, 2)
    x = x * first_conv_out
    x = _module_forward(module.final_conv, mx.concatenate((mix, x), axis=1), dtype)
    x = _cws_to_cac_mx(x, module.num_subbands)
    if module.num_target_instruments > 1:
        batch, channels, freq_bins, time_bins = x.shape
        x = x.reshape(batch, module.num_target_instruments, channels // module.num_target_instruments, freq_bins, time_bins)
    return _subband_istft(module, x, context)


def mlx_forward_mdx23c(module, raw_audio, dtype=torch.float16):
    x_mx = torch_to_mlx_input(raw_audio, dtype=dtype)
    return mlx_to_torch_mps(mlx_forward_mdx23c_mx(module, x_mx, dtype), raw_audio)
