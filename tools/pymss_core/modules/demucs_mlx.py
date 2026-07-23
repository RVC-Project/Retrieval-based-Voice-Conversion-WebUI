import math
import numpy as np
import torch

from .mlx_utils import mlx_periodic_hann_window
from .bs_roformer.mlx_attention import _gelu, _linear, _mlx_dtype, _torch_to_mlx_array, mlx_to_torch_mps
from .demucs_local import (
    LayerScale,
    MyGroupNorm,
)


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


def _reflect_pad_last(x, left=0, right=0):
    import mlx.core as mx

    parts = []
    if left > 0:
        parts.append(x[..., 1 : left + 1][..., ::-1])
    parts.append(x)
    if right > 0:
        parts.append(x[..., -right - 1 : -1][..., ::-1])
    return mx.concatenate(parts, axis=-1)


def _pad1d(x, paddings, mode="constant", value=0.0):
    import mlx.core as mx

    left, right = paddings
    if mode == "constant":
        return mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(left, right)], constant_values=value)
    if mode != "reflect":
        raise TypeError(f"unsupported Demucs padding mode for MLX full backend: {mode!r}")
    length = x.shape[-1]
    max_pad = max(left, right)
    if length <= max_pad:
        extra = max_pad - length + 1
        extra_right = min(right, extra)
        extra_left = extra - extra_right
        x = mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(extra_left, extra_right)])
        left -= extra_left
        right -= extra_right
    return _reflect_pad_last(x, left, right)


def _spectro(x, n_fft, hop, dtype):
    import mlx.core as mx

    leading = x.shape[:-1]
    length = x.shape[-1]
    flat = _reflect_pad_last(x.reshape(-1, length).astype(dtype), n_fft // 2, n_fft // 2)
    frames = 1 + (flat.shape[-1] - n_fft) // hop
    framed = mx.as_strided(flat, shape=(flat.shape[0], frames, n_fft), strides=(flat.shape[-1], hop, 1))
    spec = mx.fft.rfft(framed * _hann_window(n_fft, dtype), n=n_fft, axis=-1) / np.sqrt(n_fft)
    spec = mx.moveaxis(spec, -1, -2)
    return spec.reshape(*leading, spec.shape[-2], spec.shape[-1])


def _ispectro(z, hop, length, dtype):
    import mlx.core as mx

    leading = z.shape[:-2]
    freqs, frames_n = z.shape[-2:]
    n_fft = 2 * freqs - 2
    flat = mx.moveaxis(z.reshape(-1, freqs, frames_n), -2, -1) * np.sqrt(n_fft)
    frames = mx.fft.irfft(flat, n=n_fft, axis=-1).astype(dtype) * _hann_window(n_fft, dtype)
    full_length = n_fft + hop * (frames.shape[1] - 1)
    positions = mx.arange(n_fft)[None, :] + hop * mx.arange(frames.shape[1])[:, None]
    audio = mx.zeros((frames.shape[0], full_length), dtype=dtype).at[:, positions].add(frames)
    denom_frames = mx.broadcast_to(mx.square(_hann_window(n_fft, dtype))[None, :], (frames.shape[1], n_fft))
    denom = mx.zeros((full_length,), dtype=dtype).at[positions].add(denom_frames)
    audio = audio / mx.maximum(denom[None, :], mx.array(1e-11, dtype=dtype))
    pad = n_fft // 2
    audio = audio[..., pad : pad + length]
    return audio.reshape(*leading, audio.shape[-1])


def _demucs_spec(module, x, dtype):
    hop = module.hop_length
    n_fft = module.nfft
    le = int(math.ceil(x.shape[-1] / hop))
    pad = hop // 2 * 3
    x = _pad1d(x, (pad, pad + le * hop - x.shape[-1]), mode="reflect")
    z = _spectro(x, n_fft, hop, dtype)[..., :-1, :]
    return z[..., 2 : 2 + le]


def _demucs_ispec(module, z, length, scale, dtype):
    import mlx.core as mx

    hop = module.hop_length // (4**scale)
    z = mx.pad(z, [(0, 0)] * (z.ndim - 2) + [(0, 1), (0, 0)])
    z = mx.pad(z, [(0, 0)] * (z.ndim - 1) + [(2, 2)])
    pad = hop // 2 * 3
    le = hop * int(math.ceil(length / hop)) + 2 * pad
    x = _ispectro(z, hop, le, dtype)
    return x[..., pad : pad + length]


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


def _conv_transpose1d_ncl(conv, x, dtype):
    import mlx.core as mx

    weight = _mlx_param(conv, "weight", conv.weight, dtype).transpose(1, 2, 0)
    y = mx.conv_transpose1d(
        x.transpose(0, 2, 1),
        weight,
        stride=conv.stride[0],
        padding=conv.padding[0],
        dilation=conv.dilation[0],
        output_padding=conv.output_padding[0],
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
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
    )
    if conv.bias is not None:
        y = y + _mlx_param(conv, "bias", conv.bias, dtype)
    return y.transpose(0, 3, 1, 2)


def _conv_transpose2d_nchw(conv, x, dtype):
    import mlx.core as mx

    weight = _mlx_param(conv, "weight", conv.weight, dtype).transpose(1, 2, 3, 0)
    y = mx.conv_transpose2d(
        x.transpose(0, 2, 3, 1),
        weight,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        output_padding=conv.output_padding,
        groups=conv.groups,
    )
    if conv.bias is not None:
        y = y + _mlx_param(conv, "bias", conv.bias, dtype)
    return y.transpose(0, 3, 1, 2)


def _group_norm(module, x, dtype):
    import mlx.core as mx

    b, c = x.shape[:2]
    rest = x.shape[2:]
    groups = int(module.num_groups)
    y = x.astype(mx.float32).reshape(b, groups, c // groups, *rest)
    axes = tuple(range(2, y.ndim))
    mean = mx.mean(y, axis=axes, keepdims=True)
    var = mx.mean(mx.square(y - mean), axis=axes, keepdims=True)
    y = ((y - mean) * mx.rsqrt(var + module.eps)).reshape(x.shape).astype(x.dtype)
    if module.affine:
        shape = (1, -1) + (1,) * len(rest)
        y = y * _mlx_param(module, "weight", module.weight, dtype).reshape(*shape)
        y = y + _mlx_param(module, "bias", module.bias, dtype).reshape(*shape)
    return y


def _layer_norm(module, x, dtype):
    import mlx.core as mx

    x32 = x.astype(mx.float32)
    mean = mx.mean(x32, axis=-1, keepdims=True)
    var = mx.mean(mx.square(x32 - mean), axis=-1, keepdims=True)
    y = ((x32 - mean) * mx.rsqrt(var + module.eps)).astype(x.dtype)
    if module.elementwise_affine:
        y = y * _mlx_param(module, "weight", module.weight, dtype)
        if module.bias is not None:
            y = y + _mlx_param(module, "bias", module.bias, dtype)
    return y


def _my_group_norm(module, x, dtype):
    return _group_norm(module, x.transpose(0, 2, 1), dtype).transpose(0, 2, 1)


def _norm(module, x, dtype):
    if isinstance(module, MyGroupNorm):
        return _my_group_norm(module, x, dtype)
    if isinstance(module, torch.nn.GroupNorm):
        return _group_norm(module, x, dtype)
    if isinstance(module, torch.nn.LayerNorm):
        return _layer_norm(module, x, dtype)
    if isinstance(module, torch.nn.Identity):
        return x
    raise TypeError(f"unsupported Demucs norm for MLX full backend: {type(module).__name__}")


def _activation(module, x):
    import mlx.core as mx

    name = getattr(module, "__name__", None)
    if name == "gelu":
        return _gelu(x)
    if name == "relu":
        return mx.maximum(x, 0)
    if isinstance(module, torch.nn.GELU):
        return _gelu(x)
    if isinstance(module, torch.nn.ReLU):
        return mx.maximum(x, 0)
    if isinstance(module, torch.nn.Identity):
        return x
    raise TypeError(f"unsupported Demucs activation for MLX full backend: {type(module).__name__}")


def _glu(x, axis=1):
    import mlx.core as mx

    a, b = mx.split(x, 2, axis=axis)
    return a * mx.sigmoid(b)


def _layer_scale(module, x, dtype):
    scale = _mlx_param(module, "scale", module.scale, dtype)
    return scale * x if module.channel_last else scale[:, None] * x


def _seq(module, x, dtype):
    for child in module:
        x = _module_forward(child, x, dtype)
    return x


def _module_forward(module, x, dtype):
    if isinstance(module, torch.nn.Sequential):
        return _seq(module, x, dtype)
    if isinstance(module, torch.nn.Conv1d):
        return _conv1d_ncl(module, x, dtype)
    if isinstance(module, torch.nn.ConvTranspose1d):
        return _conv_transpose1d_ncl(module, x, dtype)
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
    if isinstance(module, (torch.nn.GroupNorm, torch.nn.LayerNorm, MyGroupNorm, torch.nn.Identity)):
        return _norm(module, x, dtype)
    if isinstance(module, (torch.nn.GELU, torch.nn.ReLU)):
        return _activation(module, x)
    if isinstance(module, torch.nn.GLU):
        return _glu(x, module.dim)
    if isinstance(module, LayerScale):
        return _layer_scale(module, x, dtype)
    raise TypeError(f"unsupported Demucs layer for MLX full backend: {type(module).__name__}")


def _dconv(module, x, dtype):
    for layer in module.layers:
        x = x + _seq(layer, x, dtype)
    return x


def _henc_layer(module, x, inject, dtype):
    import mlx.core as mx

    if not module.freq and x.ndim == 4:
        x = x.reshape(x.shape[0], -1, x.shape[-1])
    if not module.freq and x.shape[-1] % module.stride:
        x = mx.pad(x, [(0, 0), (0, 0), (0, module.stride - x.shape[-1] % module.stride)])
    y = _module_forward(module.conv, x, dtype)
    if module.empty:
        return y
    if inject is not None:
        if inject.ndim == 3 and y.ndim == 4:
            inject = inject[:, :, None]
        y = y + inject
    y = _gelu(_norm(module.norm1, y, dtype))
    if module.dconv:
        if module.freq:
            b, c, fr, t = y.shape
            y = (
                _dconv(module.dconv, y.transpose(0, 2, 1, 3).reshape(-1, c, t), dtype)
                .reshape(b, fr, c, t)
                .transpose(0, 2, 1, 3)
            )
        else:
            y = _dconv(module.dconv, y, dtype)
    return _glu(_norm(module.norm2, _module_forward(module.rewrite, y, dtype), dtype), axis=1) if module.rewrite else y


def _hdec_layer(module, x, skip, length, dtype):
    if module.freq and x.ndim == 3:
        x = x.reshape(x.shape[0], module.chin, -1, x.shape[-1])
    if module.empty:
        y = x
    else:
        x = x + skip
        y = _glu(_norm(module.norm1, _module_forward(module.rewrite, x, dtype), dtype), axis=1) if module.rewrite else x
        if module.dconv:
            if module.freq:
                b, c, fr, t = y.shape
                y = (
                    _dconv(module.dconv, y.transpose(0, 2, 1, 3).reshape(-1, c, t), dtype)
                    .reshape(b, fr, c, t)
                    .transpose(0, 2, 1, 3)
                )
            else:
                y = _dconv(module.dconv, y, dtype)
    z = _norm(module.norm2, _module_forward(module.conv_tr, y, dtype), dtype)
    if module.freq and module.pad:
        z = z[..., module.pad : -module.pad, :]
    elif not module.freq:
        z = z[..., module.pad : module.pad + length]
    return (z if module.last else _gelu(z)), y


def _create_2d_sin_embedding(d_model, height, width, dtype, max_period=10000):
    import mlx.core as mx

    half = d_model // 2
    div = mx.exp(mx.arange(0.0, half, 2) * -(math.log(max_period) / half))
    pos_w = mx.arange(0.0, width).reshape(-1, 1)
    pos_h = mx.arange(0.0, height).reshape(-1, 1)
    pe = mx.zeros((d_model, height, width), dtype=mx.float32)
    sin_w = mx.sin(pos_w * div).transpose(1, 0)[:, None, :]
    cos_w = mx.cos(pos_w * div).transpose(1, 0)[:, None, :]
    sin_h = mx.sin(pos_h * div).transpose(1, 0)[:, :, None]
    cos_h = mx.cos(pos_h * div).transpose(1, 0)[:, :, None]
    pe = pe.at[0:half:2].add(mx.broadcast_to(sin_w, (sin_w.shape[0], height, width)))
    pe = pe.at[1:half:2].add(mx.broadcast_to(cos_w, (cos_w.shape[0], height, width)))
    pe = pe.at[half::2].add(mx.broadcast_to(sin_h, (sin_h.shape[0], height, width)))
    pe = pe.at[half + 1 :: 2].add(mx.broadcast_to(cos_h, (cos_h.shape[0], height, width)))
    return pe[None].astype(dtype)


def _create_sin_embedding(length, dim, dtype, max_period=10000):
    import mlx.core as mx

    pos = mx.arange(length, dtype=mx.float32).reshape(-1, 1, 1)
    half = dim // 2
    phase = pos / (max_period ** (mx.arange(half, dtype=mx.float32).reshape(1, 1, -1) / (half - 1)))
    return mx.concatenate((mx.cos(phase), mx.sin(phase)), axis=-1).astype(dtype)


def _self_attention(mha, x, dtype):
    import mlx.core as mx

    w = _mlx_param(mha, "in_proj_weight", mha.in_proj_weight, dtype)
    b = _mlx_param(mha, "in_proj_bias", mha.in_proj_bias, dtype)
    qkv = _linear(x, w, b)
    q, k, v = mx.split(qkv, 3, axis=-1)
    heads = mha.num_heads
    head_dim = q.shape[-1] // heads
    q = q.reshape(q.shape[0], q.shape[1], heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(k.shape[0], k.shape[1], heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(v.shape[0], v.shape[1], heads, head_dim).transpose(0, 2, 1, 3)
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=head_dim**-0.5)
    out = out.transpose(0, 2, 1, 3).reshape(x.shape)
    return _linear(
        out,
        _mlx_param(mha.out_proj, "weight", mha.out_proj.weight, dtype),
        _mlx_param(mha.out_proj, "bias", mha.out_proj.bias, dtype),
    )


def _cross_attention(mha, q_in, k_in, dtype):
    import mlx.core as mx

    w = _mlx_param(mha, "in_proj_weight", mha.in_proj_weight, dtype)
    b = _mlx_param(mha, "in_proj_bias", mha.in_proj_bias, dtype)
    dim = q_in.shape[-1]
    qw, kw, vw = mx.split(w, 3, axis=0)
    qb, kb, vb = mx.split(b, 3, axis=0)
    q = _linear(q_in, qw, qb)
    k = _linear(k_in, kw, kb)
    v = _linear(k_in, vw, vb)
    heads = mha.num_heads
    head_dim = dim // heads
    q = q.reshape(q.shape[0], q.shape[1], heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(k.shape[0], k.shape[1], heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(v.shape[0], v.shape[1], heads, head_dim).transpose(0, 2, 1, 3)
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=head_dim**-0.5)
    out = out.transpose(0, 2, 1, 3).reshape(q_in.shape)
    return _linear(
        out,
        _mlx_param(mha.out_proj, "weight", mha.out_proj.weight, dtype),
        _mlx_param(mha.out_proj, "bias", mha.out_proj.bias, dtype),
    )


def _transformer_encoder_layer(module, x, dtype):
    if module.norm_first:
        x = x + _layer_scale(module.gamma_1, _self_attention(module.self_attn, _norm(module.norm1, x, dtype), dtype), dtype)
        x = x + _layer_scale(
            module.gamma_2,
            _module_forward(
                module.linear2,
                _activation(module.activation, _module_forward(module.linear1, _norm(module.norm2, x, dtype), dtype)),
                dtype,
            ),
            dtype,
        )
        return _norm(module.norm_out, x, dtype) if module.norm_out else x
    attn = _self_attention(module.self_attn, x, dtype)
    x = _norm(module.norm1, x + _layer_scale(module.gamma_1, attn, dtype), dtype)
    ffn = _module_forward(module.linear2, _activation(module.activation, _module_forward(module.linear1, x, dtype)), dtype)
    return _norm(module.norm2, x + _layer_scale(module.gamma_2, ffn, dtype), dtype)


def _cross_transformer_layer(module, q, k, dtype):
    if module.norm_first:
        attn = _cross_attention(module.cross_attn, _norm(module.norm1, q, dtype), _norm(module.norm2, k, dtype), dtype)
        x = q + _layer_scale(module.gamma_1, attn, dtype)
        ffn = _module_forward(
            module.linear2,
            _activation(module.activation, _module_forward(module.linear1, _norm(module.norm3, x, dtype), dtype)),
            dtype,
        )
        x = x + _layer_scale(module.gamma_2, ffn, dtype)
        return _norm(module.norm_out, x, dtype) if module.norm_out else x
    attn = _cross_attention(module.cross_attn, q, k, dtype)
    x = _norm(module.norm1, q + _layer_scale(module.gamma_1, attn, dtype), dtype)
    ffn = _module_forward(module.linear2, _activation(module.activation, _module_forward(module.linear1, x, dtype)), dtype)
    return _norm(module.norm2, x + _layer_scale(module.gamma_2, ffn, dtype), dtype)


def _cross_transformer(module, x, xt, dtype):
    b, c, fr, t1 = x.shape
    pos = _create_2d_sin_embedding(c, fr, t1, x.dtype, module.max_period).transpose(0, 3, 2, 1).reshape(1, t1 * fr, c)
    x = _norm(module.norm_in, x.transpose(0, 3, 2, 1).reshape(b, t1 * fr, c), dtype) + module.weight_pos_embed * pos
    b, c, t2 = xt.shape
    xt_pos = _create_sin_embedding(t2, c, xt.dtype, module.max_period).transpose(1, 0, 2)
    xt = _norm(module.norm_in_t, xt.transpose(0, 2, 1), dtype) + module.weight_pos_embed * xt_pos
    for idx in range(module.num_layers):
        if idx % 2 == module.classic_parity:
            x = _transformer_encoder_layer(module.layers[idx], x, dtype)
            xt = _transformer_encoder_layer(module.layers_t[idx], xt, dtype)
        else:
            old_x = x
            x = _cross_transformer_layer(module.layers[idx], x, xt, dtype)
            xt = _cross_transformer_layer(module.layers_t[idx], xt, old_x, dtype)
    return x.reshape(b, t1, fr, c).transpose(0, 3, 2, 1), xt.transpose(0, 2, 1)


def _std(x, axes, keepdims):
    import mlx.core as mx

    mean = mx.mean(x, axis=axes, keepdims=True)
    n = 1
    for axis in axes:
        n *= x.shape[axis]
    return mx.sqrt(mx.sum(mx.square(x - mean), axis=axes, keepdims=keepdims) / max(1, n - 1))


def _validate_supported(module):
    if module.num_subbands != 1 or not module.cac or module.wiener_iters != 0 or module.end_iters != 0:
        raise TypeError("MLX full HTDemucs supports num_subbands=1, cac=True, wiener_iters=end_iters=0")
    if any(layer.__class__.__name__ == "MultiWrap" for layer in list(module.encoder) + list(module.decoder)):
        raise TypeError("MLX full HTDemucs does not support MultiWrap/multi_freqs yet")
    if module.crosstransformer is not None and module.crosstransformer.emb != "sin":
        raise TypeError("MLX full HTDemucs supports sinusoidal transformer embeddings only")


def mlx_forward_demucs_mx(module, mix, dtype=torch.float16):
    import mlx.core as mx

    _validate_supported(module)
    if dtype not in (torch.float16, torch.float32):
        raise TypeError("MLX full HTDemucs supports torch.float16 or torch.float32 compute dtype")
    mx_dtype = _mlx_dtype(dtype)
    mix = mix.astype(mx_dtype)
    length = mix.shape[-1]
    length_pre_pad = None
    if module.use_train_segment:
        training_length = int(module.segment * module.samplerate)
        if mix.shape[-1] < training_length:
            length_pre_pad = mix.shape[-1]
            mix = mx.pad(mix, [(0, 0), (0, 0), (0, training_length - length_pre_pad)])

    z = _demucs_spec(module, mix, mx_dtype)
    b, c, fr, t = z.shape
    x = mx.stack((z.real, z.imag), axis=2).reshape(b, c * 2, fr, t)
    f_query = x.shape[2]

    mean = mx.mean(x, axis=(1, 2, 3), keepdims=True)
    std = _std(x, (1, 2, 3), keepdims=True)
    x = (x - mean) / (1e-5 + std)

    xt = mix
    meant = mx.mean(xt, axis=(1, 2), keepdims=True)
    stdt = _std(xt, (1, 2), keepdims=True)
    xt = (xt - meant) / (1e-5 + stdt)

    saved, saved_t, lengths_t = [], [], []
    for idx, encode in enumerate(module.encoder):
        skip_length = x.shape[-1]
        inject = None
        if idx < len(module.tencoder):
            lengths_t.append(xt.shape[-1])
            tenc = module.tencoder[idx]
            xt = _henc_layer(tenc, xt, None, dtype)
            if not tenc.empty:
                saved_t.append(xt)
            else:
                inject = xt
        x = _henc_layer(encode, x, inject, dtype)
        if idx == 0 and module.freq_emb is not None:
            weight = (
                _mlx_param(module.freq_emb.embedding, "weight", module.freq_emb.embedding.weight, dtype) * module.freq_emb.scale
            )
            emb = weight[: x.shape[-2]].transpose(1, 0)[None, :, :, None]
            x = x + module.freq_emb_scale * emb
        saved.append((x, skip_length))

    if module.crosstransformer:
        if module.bottom_channels:
            b, c, f, t = x.shape
            x = _conv1d_ncl(module.channel_upsampler, x.reshape(b, c, f * t), dtype).reshape(b, -1, f, t)
            xt = _conv1d_ncl(module.channel_upsampler_t, xt, dtype)
        x, xt = _cross_transformer(module.crosstransformer, x, xt, dtype)
        if module.bottom_channels:
            b, c, f, t = x.shape
            x = _conv1d_ncl(module.channel_downsampler, x.reshape(b, c, f * t), dtype).reshape(b, -1, f, t)
            xt = _conv1d_ncl(module.channel_downsampler_t, xt, dtype)

    for idx, decode in enumerate(module.decoder):
        skip, skip_length = saved.pop()
        x, pre = _hdec_layer(decode, x, skip, skip_length, dtype)
        offset = module.depth - len(module.tdecoder)
        if idx >= offset:
            tdec = module.tdecoder[idx - offset]
            length_t = lengths_t.pop()
            if tdec.empty:
                pre = pre[:, :, 0]
                xt, _ = _hdec_layer(tdec, pre, None, length_t, dtype)
            else:
                skip_t = saved_t.pop()
                xt, _ = _hdec_layer(tdec, xt, skip_t, length_t, dtype)

    stems = len(module.sources)
    x = x.reshape(b, stems, -1, f_query, t)
    x = x * std[:, None] + mean[:, None]
    x = x.reshape(b, stems, -1, 2, f_query, t).transpose(0, 1, 2, 4, 5, 3)
    zout = x[..., 0] + (1j * x[..., 1])
    out_len = int(module.segment * module.samplerate) if module.use_train_segment else length
    x_audio = _demucs_ispec(module, zout, out_len, 0, mx_dtype)

    xt = xt.reshape(b, stems, -1, out_len)
    xt = xt * stdt[:, None] + meant[:, None]
    x_audio = xt + x_audio
    if length_pre_pad:
        x_audio = x_audio[..., :length_pre_pad]
    return x_audio


def mlx_forward_demucs(module, raw_audio, dtype=torch.float16):
    x_mx = torch_to_mlx_input(raw_audio, dtype=dtype)
    return mlx_to_torch_mps(mlx_forward_demucs_mx(module, x_mx, dtype), raw_audio)
