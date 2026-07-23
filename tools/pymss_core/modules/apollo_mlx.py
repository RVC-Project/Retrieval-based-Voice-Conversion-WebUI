import torch

from .bs_roformer.mlx_attention import _mlx_dtype, _torch_to_mlx_array, mlx_to_torch_mps
from .look2hear.apollo import BSNet, ConvActNorm1d, ICB, RMSNorm


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


def _reflect_pad_last(x, pad):
    import mlx.core as mx

    if pad <= 0:
        return x
    if x.shape[-1] <= pad:
        raise ValueError("reflect padding requires input length greater than padding")
    return mx.concatenate((x[..., 1 : pad + 1][..., ::-1], x, x[..., -pad - 1 : -1][..., ::-1]), axis=-1)


def _stft(module, raw_audio, dtype):
    import mlx.core as mx

    batch_channels, length = raw_audio.shape
    n_fft = module.win
    hop = module.stride
    x = _reflect_pad_last(raw_audio.astype(dtype), n_fft // 2)
    frames = 1 + (x.shape[-1] - n_fft) // hop
    framed = mx.as_strided(x, shape=(batch_channels, frames, n_fft), strides=(x.shape[-1], hop, 1))
    window = _torch_to_mlx_array(module.window, torch.float32).astype(dtype)
    spec = mx.fft.rfft(framed * window, n=n_fft, axis=-1)
    return mx.moveaxis(spec, -1, -2), {"length": length, "n_fft": n_fft, "hop": hop, "window": window, "dtype": dtype}


def _istft(spec, context):
    import mlx.core as mx

    n_fft = context["n_fft"]
    hop = context["hop"]
    frames = mx.fft.irfft(mx.moveaxis(spec, -2, -1), n=n_fft, axis=-1).astype(context["dtype"]) * context["window"]
    frame_count = frames.shape[1]
    full_length = n_fft + hop * (frame_count - 1)
    positions = mx.arange(n_fft)[None, :] + hop * mx.arange(frame_count)[:, None]
    audio = mx.zeros((frames.shape[0], full_length), dtype=context["dtype"]).at[:, positions].add(frames)
    denom_frames = mx.broadcast_to(mx.square(context["window"])[None, :], (frame_count, n_fft))
    denom = mx.zeros((full_length,), dtype=context["dtype"]).at[positions].add(denom_frames)
    audio = audio / mx.maximum(denom[None, :], mx.array(1e-11, dtype=context["dtype"]))
    pad = n_fft // 2
    return audio[..., pad : pad + context["length"]]


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


def _rms_norm(module, x, dtype):
    import mlx.core as mx

    batch, channels, frames = x.shape
    groups = int(module.groups)
    y = x.astype(mx.float32).reshape(batch, groups, channels // groups, frames)
    y = y * mx.rsqrt(mx.mean(mx.square(y), axis=2, keepdims=True) + module.eps)
    y = y.reshape(batch, channels, frames).astype(x.dtype)
    return y * _mlx_param(module, "weight", module.weight, dtype).reshape(1, -1, 1)


def _silu(x):
    import mlx.core as mx

    return x * mx.sigmoid(x)


def _glu_channel(x):
    import mlx.core as mx

    a, b = mx.split(x, 2, axis=1)
    return a * mx.sigmoid(b)


def _module_forward(module, x, dtype):
    if isinstance(module, torch.nn.Sequential):
        for child in module:
            x = _module_forward(child, x, dtype)
        return x
    if isinstance(module, torch.nn.Conv1d):
        return _conv1d_ncl(module, x, dtype)
    if isinstance(module, RMSNorm):
        return _rms_norm(module, x, dtype)
    if isinstance(module, torch.nn.SiLU):
        return _silu(x)
    if isinstance(module, torch.nn.GLU):
        return _glu_channel(x)
    if isinstance(module, ConvActNorm1d):
        return _conv_act_norm(module, x, dtype)
    if isinstance(module, ICB):
        return _module_forward(module.blocks, x, dtype)
    if isinstance(module, BSNet):
        return _bsnet(module, x, dtype)
    raise TypeError(f"unsupported Apollo layer for MLX full backend: {type(module).__name__}")


def _conv_act_norm(module, x, dtype):
    y = _conv1d_ncl(module.conv[0], x, dtype)
    y = _rms_norm(module.conv[1], y, dtype)
    y = _conv1d_ncl(module.conv[2], y, dtype)
    y = _silu(y)
    y = _conv1d_ncl(module.conv[4], y, dtype)
    if module.causal:
        y = y[..., : -module.kernel + 1]
    return x + y


def _apply_rope(module, x, dtype):
    import mlx.core as mx

    seq_len = x.shape[-2]
    cos = _torch_to_mlx_array(module.cos_freq[:seq_len], dtype).reshape(1, 1, seq_len, -1)
    sin = _torch_to_mlx_array(module.sin_freq[:seq_len], dtype).reshape(1, 1, seq_len, -1)
    even, odd = x[..., 0::2], x[..., 1::2]
    cos_e = cos[..., 0::2]
    sin_e = sin[..., 0::2]
    out = mx.zeros_like(x)
    out = out.at[..., 0::2].add(even * cos_e - odd * sin_e)
    out = out.at[..., 1::2].add(odd * cos_e + even * sin_e)
    return out


def _roformer(module, x, dtype):
    import mlx.core as mx

    batch, _, frames = x.shape
    x_norm = _rms_norm(module.input_norm, x, dtype)
    qkv = _conv1d_ncl(module.weight, x_norm, dtype)
    qkv = qkv.reshape(batch, module.num_head, module.hidden_size * 3, frames).transpose(0, 1, 3, 2)
    q, k, v = mx.split(qkv, 3, axis=-1)
    q = _apply_rope(module, q, dtype)
    k = _apply_rope(module, k, dtype)
    attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=module.hidden_size**-0.5, mask=None)
    out = attn.transpose(0, 1, 3, 2).reshape(batch, -1, frames)
    out = _conv1d_ncl(module.output, out, dtype) + x

    hidden = _rms_norm(module.MLP[0], out, dtype)
    hidden = _conv1d_ncl(module.MLP[1], hidden, dtype)
    hidden = _silu(hidden)
    gate, z = mx.split(hidden, 2, axis=1)
    return out + _conv1d_ncl(module.MLP_output, _silu(gate) * z, dtype)


def _bsnet(module, x, dtype):
    batch, bands, channels, frames = x.shape
    band = x.transpose(0, 3, 2, 1).reshape(batch * frames, channels, bands)
    band = _roformer(module.band_net, band, dtype)
    band = band.reshape(batch, frames, channels, bands).transpose(0, 3, 2, 1)
    seq = _module_forward(module.seq_net, band.reshape(batch * bands, channels, frames), dtype)
    return seq.reshape(batch, bands, channels, frames)


def _feature_extractor(module, raw_audio, dtype):
    import mlx.core as mx

    mx_dtype = _mlx_dtype(dtype)
    batch, channels, samples = raw_audio.shape
    spec, _ = _stft(module, raw_audio.reshape(batch * channels, samples), mx_dtype)
    features = []
    powers = []
    band_index = 0
    for width, bn in zip(module.band_width, module.BN):
        sub = spec[:, band_index : band_index + width]
        power = mx.sqrt(mx.sum(mx.square(sub.real) + mx.square(sub.imag), axis=1, keepdims=True) + module.eps)
        norm = sub / power
        inp = mx.concatenate((norm.real, norm.imag, mx.log(power)), axis=1)
        features.append(_module_forward(bn, inp.astype(mx_dtype), dtype))
        powers.append(power)
        band_index += width
    return mx.stack(features, axis=1), spec


def _estimate_spec(module, feature, batch_channels, dtype):
    import mlx.core as mx

    specs = []
    for band_feature, output, width in zip(mx.split(feature, feature.shape[1], axis=1), module.output, module.band_width):
        band_feature = band_feature[:, 0]
        ri = _module_forward(output, band_feature, dtype).reshape(batch_channels, 2, width, -1)
        specs.append(ri[:, 0] + (1j * ri[:, 1]))
    return mx.concatenate(specs, axis=1)


def mlx_forward_apollo_mx(module, raw_audio, dtype=torch.float16):
    if dtype not in (torch.float16, torch.float32):
        raise TypeError("MLX full Apollo supports torch.float16 or torch.float32 compute dtype")
    mx_dtype = _mlx_dtype(dtype)
    raw_audio = raw_audio.astype(mx_dtype)
    batch, channels, samples = raw_audio.shape
    feature, _ = _feature_extractor(module, raw_audio, dtype)
    for block in module.net:
        feature = _bsnet(block, feature, dtype)
    est_spec = _estimate_spec(module, feature, batch * channels, dtype)
    return _istft(
        est_spec,
        {
            "length": samples,
            "n_fft": module.win,
            "hop": module.stride,
            "window": _torch_to_mlx_array(module.window, torch.float32).astype(mx_dtype),
            "dtype": mx_dtype,
        },
    ).reshape(batch, channels, -1)


def mlx_forward_apollo(module, raw_audio, dtype=torch.float16):
    x_mx = torch_to_mlx_input(raw_audio, dtype=dtype)
    return mlx_to_torch_mps(mlx_forward_apollo_mx(module, x_mx, dtype), raw_audio)
