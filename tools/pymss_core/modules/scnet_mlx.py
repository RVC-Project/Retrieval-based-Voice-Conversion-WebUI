import math
import numpy as np
import torch

from .mlx_utils import mlx_periodic_hann_window
from .bs_roformer.mlx_attention import _gelu, _linear, _mlx_dtype, _torch_to_mlx_array, mlx_to_torch_mps
from .scnet.scnet import Swish


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


def _stft_scnet(module, raw_audio, dtype):
    import mlx.core as mx

    n_fft = int(module.stft_config["n_fft"])
    hop = int(module.stft_config["hop_length"])
    win_length = int(module.stft_config["win_length"])
    normalized = bool(module.stft_config.get("normalized", False))
    center = bool(module.stft_config.get("center", True))
    leading_shape = raw_audio.shape[:-1]
    length = raw_audio.shape[-1]
    x = raw_audio.reshape(-1, length).astype(dtype)
    if center:
        x = _reflect_pad_last(x, n_fft // 2)
    frames = 1 + (x.shape[-1] - n_fft) // hop
    framed = mx.as_strided(x, shape=(x.shape[0], frames, n_fft), strides=(x.shape[-1], hop, 1))
    window = mx.ones((win_length,), dtype=dtype)
    if win_length < n_fft:
        left = (n_fft - win_length) // 2
        window = mx.pad(window, [(left, n_fft - win_length - left)])
    spec = mx.fft.rfft(framed * window, n=n_fft, axis=-1)
    if normalized:
        spec = spec / np.sqrt(n_fft)
    spec = mx.moveaxis(spec, -1, -2)
    return spec.reshape(*leading_shape, spec.shape[-2], spec.shape[-1]), {
        "n_fft": n_fft,
        "hop": hop,
        "window": window,
        "normalized": normalized,
        "center": center,
        "dtype": dtype,
    }


def _istft_scnet(module, spec, context, length):
    import mlx.core as mx

    leading_shape = spec.shape[:-2]
    freqs, frames_n = spec.shape[-2:]
    n_fft = context["n_fft"]
    hop = context["hop"]
    complex_stft = mx.moveaxis(spec.reshape(-1, freqs, frames_n), -2, -1)
    if context["normalized"]:
        complex_stft = complex_stft * np.sqrt(n_fft)
    frames = mx.fft.irfft(complex_stft, n=n_fft, axis=-1).astype(context["dtype"]) * context["window"]
    full_length = n_fft + hop * (frames.shape[1] - 1)
    positions = mx.arange(n_fft)[None, :] + hop * mx.arange(frames.shape[1])[:, None]
    audio = mx.zeros((frames.shape[0], full_length), dtype=context["dtype"]).at[:, positions].add(frames)
    denom_frames = mx.broadcast_to(mx.square(context["window"])[None, :], (frames.shape[1], n_fft))
    denom = mx.zeros((full_length,), dtype=context["dtype"]).at[positions].add(denom_frames)
    audio = audio / mx.maximum(denom[None, :], mx.array(1e-11, dtype=context["dtype"]))
    if context["center"]:
        pad = n_fft // 2
        audio = audio[..., pad : pad + length]
    else:
        audio = audio[..., :length]
    return audio.reshape(*leading_shape, audio.shape[-1])


def _conv_padding(conv):
    padding = conv.padding
    if isinstance(padding, tuple):
        return padding[0] if len(padding) == 1 else padding
    return padding


def _conv1d_ncl(conv, x, dtype):
    import mlx.core as mx

    weight = _mlx_param(conv, "weight", conv.weight, dtype).transpose(0, 2, 1)
    y = mx.conv1d(
        x.transpose(0, 2, 1),
        weight,
        stride=conv.stride[0],
        padding=conv.padding[0],
        dilation=conv.dilation[0],
        groups=conv.groups,
    )
    if conv.bias is not None:
        y = y + _mlx_param(conv, "bias", conv.bias, dtype)
    return y.transpose(0, 2, 1)


def _conv2d_nchw(conv, x, dtype):
    import mlx.core as mx

    weight = _mlx_param(conv, "weight", conv.weight, dtype).transpose(0, 2, 3, 1)
    y = mx.conv2d(
        x.transpose(0, 2, 3, 1),
        weight,
        stride=conv.stride,
        padding=_conv_padding(conv),
        dilation=conv.dilation,
        groups=conv.groups,
    )
    if conv.bias is not None:
        y = y + _mlx_param(conv, "bias", conv.bias, dtype)
    return y.transpose(0, 3, 1, 2)


def _conv_transpose2d_nchw(conv, x, dtype):
    import mlx.core as mx

    if conv.groups != 1:
        raise TypeError("MLX SCNet ConvTranspose2d currently supports groups=1 only")
    weight = _mlx_param(conv, "weight", conv.weight, dtype).transpose(1, 2, 3, 0)
    y = mx.conv_transpose2d(
        x.transpose(0, 2, 3, 1),
        weight,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
    )
    if conv.bias is not None:
        y = y + _mlx_param(conv, "bias", conv.bias, dtype)
    return y.transpose(0, 3, 1, 2)


def _group_norm(module, x, dtype):
    import mlx.core as mx

    b, c = x.shape[:2]
    rest = x.shape[2:]
    y = x.astype(mx.float32).reshape(b, module.num_groups, c // module.num_groups, *rest)
    axes = tuple(range(2, y.ndim))
    mean = mx.mean(y, axis=axes, keepdims=True)
    var = mx.mean(mx.square(y - mean), axis=axes, keepdims=True)
    y = ((y - mean) * mx.rsqrt(var + module.eps)).reshape(x.shape).astype(x.dtype)
    if module.affine:
        shape = (1, -1) + (1,) * len(rest)
        y = y * _mlx_param(module, "weight", module.weight, dtype).reshape(*shape)
        y = y + _mlx_param(module, "bias", module.bias, dtype).reshape(*shape)
    return y


def _activation(module, x):
    import mlx.core as mx

    if isinstance(module, torch.nn.GELU):
        return _gelu(x)
    if isinstance(module, torch.nn.ReLU):
        return mx.maximum(x, 0)
    if isinstance(module, Swish):
        return x * mx.sigmoid(x)
    if isinstance(module, torch.nn.Identity):
        return x
    raise TypeError(f"unsupported SCNet activation for MLX full backend: {type(module).__name__}")


def _glu(x, axis):
    import mlx.core as mx

    a, b = mx.split(x, 2, axis=axis)
    return a * mx.sigmoid(b)


def _seq(module, x, dtype):
    for child in module:
        x = _module_forward(child, x, dtype)
    return x


def _module_forward(module, x, dtype):
    if isinstance(module, torch.nn.Sequential):
        return _seq(module, x, dtype)
    if isinstance(module, torch.nn.Conv1d):
        return _conv1d_ncl(module, x, dtype)
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
    if isinstance(module, torch.nn.GroupNorm):
        return _group_norm(module, x, dtype)
    if isinstance(module, torch.nn.GLU):
        return _glu(x, axis=module.dim)
    if isinstance(module, (torch.nn.GELU, torch.nn.ReLU, Swish, torch.nn.Identity)):
        return _activation(module, x)
    raise TypeError(f"unsupported SCNet layer for MLX full backend: {type(module).__name__}")


def _convolution_module(module, x, dtype):
    for layer in module.layers:
        x = x + _seq(layer, x, dtype)
    return x


def _sdlayer(module, x, dtype):
    import mlx.core as mx

    fr = x.shape[2]
    low = math.ceil(fr * module.SR_low)
    mid = math.ceil(fr * (module.SR_low + module.SR_mid))
    splits = [(0, low), (low, mid), (mid, fr)]
    outputs, original_lengths = [], []
    for conv, stride, kernel, (start, end) in zip(module.convs, module.strides, module.kernels, splits):
        extracted = x[:, :, start:end, :]
        original_lengths.append(end - start)
        total_padding = kernel - stride if stride == 1 else (stride - extracted.shape[2] % stride) % stride
        pad_left = total_padding // 2
        padded = mx.pad(extracted, [(0, 0), (0, 0), (pad_left, total_padding - pad_left), (0, 0)])
        outputs.append(_conv2d_nchw(conv, padded, dtype))
    return outputs, original_lengths


def _sdblock(module, x, dtype):
    import mlx.core as mx

    bands, original_lengths = _sdlayer(module.SDlayer, x, dtype)
    outs = []
    for conv, band in zip(module.conv_modules, bands):
        b, c, f, t = band.shape
        out = _convolution_module(conv, band.transpose(0, 2, 1, 3).reshape(b * f, c, t), dtype)
        out = _gelu(out.reshape(b, f, c, t).transpose(0, 2, 1, 3))
        outs.append(out)
    lengths = [band.shape[-2] for band in outs]
    full_band = mx.concatenate(outs, axis=2)
    return _conv2d_nchw(module.globalconv, full_band, dtype), full_band, lengths, original_lengths


def _rnn_forward_lstm(rnn, x, dtype):
    import mlx.core as mx

    def params(suffix):
        return {
            "w_ih": _mlx_param(rnn, f"weight_ih_l0{suffix}", getattr(rnn, f"weight_ih_l0{suffix}"), dtype),
            "w_hh": _mlx_param(rnn, f"weight_hh_l0{suffix}", getattr(rnn, f"weight_hh_l0{suffix}"), dtype),
            "b_ih": _mlx_param(rnn, f"bias_ih_l0{suffix}", getattr(rnn, f"bias_ih_l0{suffix}"), dtype),
            "b_hh": _mlx_param(rnn, f"bias_hh_l0{suffix}", getattr(rnn, f"bias_hh_l0{suffix}"), dtype),
        }

    def run(p, reverse=False):
        steps = range(x.shape[1] - 1, -1, -1) if reverse else range(x.shape[1])
        h = mx.zeros((x.shape[0], rnn.hidden_size), dtype=x.dtype)
        c = mx.zeros_like(h)
        outs = []
        for t in steps:
            gates = _linear(x[:, t], p["w_ih"], p["b_ih"]) + _linear(h, p["w_hh"], p["b_hh"])
            i, f, g, o = mx.split(gates, 4, axis=-1)
            i, f, o = mx.sigmoid(i), mx.sigmoid(f), mx.sigmoid(o)
            c = f * c + i * mx.tanh(g)
            h = o * mx.tanh(c)
            outs.append(h)
        if reverse:
            outs.reverse()
        return mx.stack(outs, axis=1)

    if rnn.num_layers != 1 or not rnn.batch_first:
        raise TypeError("MLX SCNet LSTM supports one-layer batch_first LSTMs only")
    forward = run(params(""))
    if not rnn.bidirectional:
        return forward
    return mx.concatenate((forward, run(params("_reverse"), reverse=True)), axis=-1)


def _dual_path_rnn(module, x, dtype):
    b, c, f, t = x.shape
    y = _group_norm(module.norm_layers[0], x, dtype).transpose(0, 3, 2, 1).reshape(b * t, f, c)
    y = _rnn_forward_lstm(module.lstm_layers[0], y, dtype)
    y = _linear(
        y,
        _mlx_param(module.linear_layers[0], "weight", module.linear_layers[0].weight, dtype),
        _mlx_param(module.linear_layers[0], "bias", module.linear_layers[0].bias, dtype),
    )
    x = y.reshape(b, t, f, c).transpose(0, 3, 2, 1) + x

    y = _group_norm(module.norm_layers[1], x, dtype).transpose(0, 2, 1, 3).reshape(b * f, c, t).transpose(0, 2, 1)
    y = _rnn_forward_lstm(module.lstm_layers[1], y, dtype)
    y = _linear(
        y,
        _mlx_param(module.linear_layers[1], "weight", module.linear_layers[1].weight, dtype),
        _mlx_param(module.linear_layers[1], "bias", module.linear_layers[1].bias, dtype),
    )
    return y.transpose(0, 2, 1).reshape(b, f, c, t).transpose(0, 2, 1, 3) + x


def _feature_conversion(module, x):
    import mlx.core as mx

    x = x.astype(mx.float32)
    if module.inverse:
        half = module.channels // 2
        return mx.fft.irfft(x[:, :half] + (1j * x[:, half:]), n=(x.shape[3] - 1) * 2, axis=3, norm="ortho")
    x = mx.fft.rfft(x, axis=3, norm="ortho")
    return mx.concatenate((x.real, x.imag), axis=1)


def _separation_net(module, x, dtype):
    for dp_module, feature_conversion in zip(module.dp_modules, module.feature_conversion):
        x = _feature_conversion(feature_conversion, _dual_path_rnn(dp_module, x, dtype))
    return x


def _crop_center(skip, target):
    h, w = target.shape[2], target.shape[3]
    dh = (skip.shape[2] - h) // 2
    dw = (skip.shape[3] - w) // 2
    return skip[:, :, dh : dh + h, dw : dw + w]


def _fusion_layer(module, x, skip, dtype):
    import mlx.core as mx

    if skip is not None:
        x = x + skip
    return _glu(_conv2d_nchw(module.conv, mx.concatenate((x, x), axis=1), dtype), axis=1)


def _sulayer(module, x, lengths, origin_lengths, dtype):
    import mlx.core as mx

    ranges = [(0, lengths[0]), (lengths[0], lengths[0] + lengths[1]), (lengths[0] + lengths[1], None)]
    outs = []
    for idx, (convtr, (start, end)) in enumerate(zip(module.convtrs, ranges)):
        out = _conv_transpose2d_nchw(convtr, x[:, :, start:end, :], dtype)
        dist = abs(origin_lengths[idx] - out.shape[2]) // 2
        outs.append(out[:, :, dist : dist + origin_lengths[idx], :])
    return mx.concatenate(outs, axis=2)


def mlx_forward_scnet_mx(module, raw_audio, dtype=torch.float16):
    import mlx.core as mx

    if dtype not in (torch.float16, torch.float32):
        raise TypeError("MLX full SCNet supports torch.float16 or torch.float32 compute dtype")
    mx_dtype = _mlx_dtype(dtype)
    x = raw_audio.astype(mx_dtype)
    batch = x.shape[0]
    padding = module.hop_length - x.shape[-1] % module.hop_length
    if (x.shape[-1] + padding) // module.hop_length % 2 == 0:
        padding += module.hop_length
    x = mx.pad(x, [(0, 0), (0, 0), (0, padding)])

    length = x.shape[-1]
    spec, context = _stft_scnet(module, x.reshape(-1, length), mx_dtype)
    ri = mx.stack((spec.real, spec.imag), axis=-1)
    x = ri.transpose(0, 3, 1, 2).reshape(
        ri.shape[0] // module.audio_channels, ri.shape[3] * module.audio_channels, ri.shape[1], ri.shape[2]
    )
    _, _, freq_bins, time_bins = x.shape

    saved = []
    for sd_layer in module.encoder:
        x, skip, lengths, original_lengths = _sdblock(sd_layer, x, dtype)
        saved.append((skip, lengths, original_lengths))

    x = _separation_net(module.separation_net, x, dtype)

    for decoder in module.decoder:
        fusion_layer, su_layer = decoder
        skip, lengths, original_lengths = saved.pop()
        x = _sulayer(su_layer, _fusion_layer(fusion_layer, x, skip, dtype), lengths, original_lengths, dtype)

    n = module.dims[0]
    x = x.reshape(batch, n, -1, freq_bins, time_bins).reshape(-1, 2, freq_bins, time_bins).transpose(0, 2, 3, 1)
    spec_out = x[..., 0] + (1j * x[..., 1])
    audio = _istft_scnet(module, spec_out, context, length)
    audio = audio.reshape(batch, len(module.sources), module.audio_channels, -1)
    return audio[:, :, :, :-padding] if padding > 0 else audio


def mlx_forward_scnet(module, raw_audio, dtype=torch.float16):
    x_mx = torch_to_mlx_input(raw_audio, dtype=dtype)
    return mlx_to_torch_mps(mlx_forward_scnet_mx(module, x_mx, dtype), raw_audio)
