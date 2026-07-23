import numpy as np
import torch

from ..mlx_utils import mlx_periodic_hann_window
from .bands import contiguous_dim_groups, dim_input_offsets
from .bs_roformer_hyperace import BSRoformerHyperACE
from . import hyperace_segm
from .mel_band_roformer import MelBandRoformer
from .mlx_attention import (
    _COMPUTE_DTYPE,
    _gelu,
    _linear,
    _mlx_attention,
    _mlx_dtype,
    _mlx_feed_forward,
    _mlx_output_norm,
    _rms_norm,
    _torch_to_mlx_array,
    mlx_to_torch_mps,
)


def torch_to_mlx_input(tensor, dtype=_COMPUTE_DTYPE):
    import mlx.core as mx

    return mx.array(tensor.detach().to(dtype=dtype).cpu().numpy())


def _cache_key(params, dtype):
    return tuple(None if p is None else (p.data_ptr(), p._version, tuple(p.shape), dtype) for p in params)


def _is_identity(module):
    return isinstance(module, torch.nn.Identity)


def _hann_window(length, dtype):
    return mlx_periodic_hann_window(length, dtype)


def _reflect_pad_last(x, pad):
    import mlx.core as mx

    if pad <= 0:
        return x
    if x.shape[-1] <= pad:
        raise ValueError("reflect padding requires input length greater than padding")
    left = x[..., 1 : pad + 1][..., ::-1]
    right = x[..., -pad - 1 : -1][..., ::-1]
    return mx.concatenate((left, x, right), axis=-1)


def _stft_roformer(module, raw_audio, dtype):
    import mlx.core as mx

    if raw_audio.ndim == 2:
        raw_audio = raw_audio[:, None, :]

    batch, channels, audio_length = raw_audio.shape
    if (module.stereo and channels != 2) or (not module.stereo and channels != 1):
        raise ValueError("raw_audio channel count does not match RoFormer stereo setting")

    n_fft = int(module.stft_kwargs["n_fft"])
    hop = int(module.stft_kwargs["hop_length"])
    win_length = int(module.stft_kwargs["win_length"])
    normalized = bool(module.stft_kwargs.get("normalized", False))
    stft_audio = raw_audio.reshape(batch * channels, audio_length).astype(dtype)
    stft_audio = _reflect_pad_last(stft_audio, n_fft // 2)

    frames = 1 + (stft_audio.shape[-1] - n_fft) // hop
    framed = mx.as_strided(
        stft_audio,
        shape=(batch * channels, frames, n_fft),
        strides=(stft_audio.shape[-1], hop, 1),
    )
    window = _hann_window(win_length, dtype)
    if win_length < n_fft:
        left = (n_fft - win_length) // 2
        right = n_fft - win_length - left
        window = mx.pad(window, [(left, right)])
    elif win_length > n_fft:
        raise ValueError("MLX RoFormer STFT does not support win_length > n_fft")

    stft = mx.fft.rfft(framed * window, n=n_fft, axis=-1)
    if normalized:
        stft = stft / np.sqrt(n_fft)
    stft = mx.moveaxis(stft, -1, -2)
    stft_repr = mx.stack((stft.real, stft.imag), axis=-1)
    freq_bins = stft_repr.shape[-3]
    stft_repr = stft_repr.reshape(batch, channels, freq_bins, frames, 2)
    stft_repr = mx.transpose(stft_repr, (0, 2, 1, 3, 4)).reshape(batch, freq_bins * channels, frames, 2)
    context = {
        "batch": batch,
        "channels": channels,
        "freq_bins": freq_bins,
        "audio_length": audio_length,
        "window": window,
        "n_fft": n_fft,
        "hop": hop,
        "normalized": normalized,
        "dtype": dtype,
    }
    return stft_repr, context


def _istft_roformer(module, stft_repr, context, length):
    import mlx.core as mx

    b, n, _, t, _ = stft_repr.shape
    channels = context["channels"]
    freq_bins = context["freq_bins"]
    n_fft = context["n_fft"]
    hop = context["hop"]
    dtype = context["dtype"]
    normalized = context["normalized"]
    stft_repr = stft_repr.reshape(b, n, freq_bins, channels, t, 2)
    stft_repr = mx.transpose(stft_repr, (0, 1, 3, 2, 4, 5)).reshape(b * n * channels, freq_bins, t, 2)
    complex_stft = stft_repr[..., 0] + (1j * stft_repr[..., 1])
    if getattr(module, "zero_dc", False):
        complex_stft = complex_stft.at[:, 0, :].set(0)
    complex_stft = mx.moveaxis(complex_stft, -2, -1)
    if normalized:
        complex_stft = complex_stft * np.sqrt(n_fft)
    frames = mx.fft.irfft(complex_stft, n=n_fft, axis=-1).astype(dtype)
    frames = frames * context["window"]

    batch_channels = frames.shape[0]
    frame_count = frames.shape[1]
    full_length = n_fft + hop * (frame_count - 1)
    positions = mx.arange(n_fft)[None, :] + hop * mx.arange(frame_count)[:, None]
    audio = mx.zeros((batch_channels, full_length), dtype=dtype).at[:, positions].add(frames)
    denom_frames = mx.broadcast_to(mx.square(context["window"])[None, :], (frame_count, n_fft))
    denom = mx.zeros((full_length,), dtype=dtype).at[positions].add(denom_frames)
    audio = audio / mx.maximum(denom[None, :], mx.array(1e-11, dtype=dtype))
    pad = n_fft // 2
    if length is None:
        audio = audio[..., pad:-pad] if pad > 0 else audio
    else:
        audio = audio[..., pad : pad + length]
    audio = audio.reshape(context["batch"], n, channels, audio.shape[-1])
    return audio[:, 0] if n == 1 else audio


def _band_split_cache(module, dtype):
    cache = getattr(module, "_pymss_mlx_full_band_split_cache", None)
    params = []
    for to_feature in module.band_split.to_features:
        norm, linear = to_feature
        params.extend((norm.gamma, linear.weight, linear.bias))
    key = (tuple(module.band_split.dim_inputs), _cache_key(params, dtype))
    if cache is not None and cache.get("key") == key:
        return cache

    groups = []
    for start, end, dim_in in contiguous_dim_groups(module.band_split.dim_inputs):
        norms = [module.band_split.to_features[i][0] for i in range(start, end)]
        linears = [module.band_split.to_features[i][1] for i in range(start, end)]
        groups.append(
            {
                "start": start,
                "end": end,
                "dim_in": dim_in,
                "offset_start": module.band_split._dim_offsets[start],
                "offset_end": module.band_split._dim_offsets[end],
                "gamma": _torch_to_mlx_array(torch.stack([norm.gamma for norm in norms], dim=0), dtype),
                "weight": _torch_to_mlx_array(torch.stack([linear.weight for linear in linears], dim=0), dtype),
                "bias": None
                if linears[0].bias is None
                else _torch_to_mlx_array(torch.stack([linear.bias for linear in linears], dim=0), dtype),
            }
        )

    cache = {"key": key, "groups": groups}
    module._pymss_mlx_full_band_split_cache = cache
    return cache


def _grouped_linear(x, weight, bias):
    import mlx.core as mx

    out = mx.einsum("...gi,goi->...go", x, weight)
    return out if bias is None else out + bias


def _band_split(module, x, dtype):
    import mlx.core as mx

    outs = []
    for group in _band_split_cache(module, dtype)["groups"]:
        group_x = x[..., group["offset_start"] : group["offset_end"]]
        group_x = group_x.reshape(*group_x.shape[:-1], group["end"] - group["start"], group["dim_in"])
        group_x = _rms_norm(group_x, group["gamma"])
        outs.append(_grouped_linear(group_x, group["weight"], group["bias"]))
    return mx.concatenate(outs, axis=-2)


def _transformer(module, x, dtype):
    for attn, ff in module.layers:
        x = _mlx_attention(attn, x, dtype) + x
        x = _mlx_feed_forward(ff, x, dtype) + x
    return _mlx_output_norm(module.norm, x, dtype)


def _final_norm(module, x, dtype):
    if _is_identity(module.final_norm):
        return x
    return _rms_norm(x, _torch_to_mlx_array(module.final_norm.gamma, dtype))


def _mask_estimator_layers(mlp_with_glu):
    layers = []
    mlp, glu = mlp_with_glu
    if not isinstance(glu, torch.nn.GLU):
        raise TypeError("MLX RoFormer mask estimator expects nn.GLU")
    for layer in mlp:
        if isinstance(layer, torch.nn.Linear):
            layers.append(("linear", layer))
        elif isinstance(layer, torch.nn.Tanh):
            layers.append(("tanh", None))
        else:
            raise TypeError(f"unsupported MLX RoFormer mask estimator layer: {type(layer).__name__}")
    return tuple(layers)


def _mask_estimator_cache(estimator, dtype):
    cache = getattr(estimator, "_pymss_mlx_full_mask_cache", None)
    params = []
    for mlp_with_glu in estimator.to_freqs:
        for kind, layer in _mask_estimator_layers(mlp_with_glu):
            if kind == "linear":
                params.extend((layer.weight, layer.bias))
    key = (tuple(estimator.dim_inputs), _cache_key(params, dtype))
    if cache is not None and cache.get("key") == key:
        return cache

    band_layers = []
    for mlp_with_glu in estimator.to_freqs:
        layers = []
        for kind, layer in _mask_estimator_layers(mlp_with_glu):
            if kind == "tanh":
                layers.append(("tanh", None, None))
            else:
                layers.append(
                    (
                        "linear",
                        _torch_to_mlx_array(layer.weight, dtype),
                        None if layer.bias is None else _torch_to_mlx_array(layer.bias, dtype),
                    )
                )
        band_layers.append(tuple(layers))
    cache = {"key": key, "band_layers": tuple(band_layers)}
    estimator._pymss_mlx_full_mask_cache = cache
    return cache


def _glu(x, axis=-1):
    import mlx.core as mx

    a, b = mx.split(x, 2, axis=axis)
    return a * (1 / (1 + mx.exp(-b)))


def _mask_estimator(estimator, x, dtype):
    import mlx.core as mx

    outs = []
    for band_index, layers in enumerate(_mask_estimator_cache(estimator, dtype)["band_layers"]):
        group_x = x[:, :, band_index, :]
        for kind, weight, bias in layers:
            if kind == "tanh":
                group_x = mx.tanh(group_x)
            else:
                group_x = _linear(group_x, weight, bias)
        outs.append(_glu(group_x, axis=-1))
    return mx.concatenate(outs, axis=-1)


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


def _silu(x):
    import mlx.core as mx

    return x * mx.sigmoid(x)


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

    weight = _mlx_param(conv, "weight", conv.weight, dtype)
    weight = mx.transpose(weight, (0, 2, 3, 1))
    x_nhwc = mx.transpose(x, (0, 2, 3, 1))
    y = mx.conv2d(
        x_nhwc,
        weight,
        stride=conv.stride,
        padding=_conv_padding(conv),
        dilation=conv.dilation,
        groups=conv.groups,
    )
    if conv.bias is not None:
        y = y + _mlx_param(conv, "bias", conv.bias, dtype)
    return mx.transpose(y, (0, 3, 1, 2))


def _instance_norm2d(module, x, dtype):
    import mlx.core as mx

    x32 = x.astype(mx.float32)
    mean = mx.mean(x32, axis=(2, 3), keepdims=True)
    var = mx.mean(mx.square(x32 - mean), axis=(2, 3), keepdims=True)
    x = ((x32 - mean) * mx.rsqrt(var + module.eps)).astype(x.dtype)
    if module.affine:
        weight = _mlx_param(module, "weight", module.weight, dtype).reshape(1, -1, 1, 1)
        bias = _mlx_param(module, "bias", module.bias, dtype).reshape(1, -1, 1, 1)
        x = x * weight + bias
    return x


def _conv_block(module, x, dtype):
    x = _conv2d_nchw(module.conv, x, dtype)
    x = _instance_norm2d(module.bn, x, dtype)
    return x if isinstance(module.act, torch.nn.Identity) else _silu(x)


def _dsconv_block(module, x, dtype):
    x = _conv2d_nchw(module.dwconv, x, dtype)
    x = _conv2d_nchw(module.pwconv, x, dtype)
    x = _instance_norm2d(module.bn, x, dtype)
    return x if isinstance(module.act, torch.nn.Identity) else _silu(x)


def _resize_positions(in_size, out_size):
    import mlx.core as mx

    scale = in_size / out_size
    pos = (mx.arange(out_size, dtype=mx.float32) + 0.5) * scale - 0.5
    lower = mx.floor(pos)
    upper = lower + 1
    weight = pos - lower
    lower = mx.clip(lower, 0, in_size - 1).astype(mx.int32)
    upper = mx.clip(upper, 0, in_size - 1).astype(mx.int32)
    return lower, upper, weight


def _resize_bilinear_nchw(x, size):
    import mlx.core as mx

    out_h, out_w = int(size[0]), int(size[1])
    in_h, in_w = x.shape[2], x.shape[3]
    if in_h == out_h and in_w == out_w:
        return x
    y0, y1, wy = _resize_positions(in_h, out_h)
    x0, x1, wx = _resize_positions(in_w, out_w)
    v00 = mx.take(mx.take(x, y0, axis=2), x0, axis=3)
    v01 = mx.take(mx.take(x, y0, axis=2), x1, axis=3)
    v10 = mx.take(mx.take(x, y1, axis=2), x0, axis=3)
    v11 = mx.take(mx.take(x, y1, axis=2), x1, axis=3)
    wy = wy.reshape(1, 1, out_h, 1)
    wx = wx.reshape(1, 1, 1, out_w)
    return v00 * (1 - wy) * (1 - wx) + v01 * (1 - wy) * wx + v10 * wy * (1 - wx) + v11 * wy * wx


def _seq(module, x, dtype):
    for child in module:
        x = _segm_module(child, x, dtype)
    return x


def _ds_bottleneck(module, x, dtype):
    y = _dsconv_block(module.dsconv2, _dsconv_block(module.dsconv1, x, dtype), dtype)
    return x + y if module.shortcut else y


def _ds_c3k(module, x, dtype):
    import mlx.core as mx

    return _conv_block(
        module.cv3,
        mx.concatenate((_seq(module.m, _conv_block(module.cv1, x, dtype), dtype), _conv_block(module.cv2, x, dtype)), axis=1),
        dtype,
    )


def _ds_c3k2(module, x, dtype):
    return _conv_block(module.cv2, _ds_c3k(module.m, _conv_block(module.cv1, x, dtype), dtype), dtype)


def _adaptive_hyperedge_generation(module, x, dtype):
    import mlx.core as mx

    b, n, c = x.shape
    avg_pool = mx.mean(x, axis=1)
    max_pool = mx.max(x, axis=1)
    context = mx.concatenate((avg_pool, max_pool), axis=1)
    proto = _mlx_param(module, "global_proto", module.global_proto, dtype)[None]
    mapped = _linear(context, _mlx_param(module.context_mapper, "weight", module.context_mapper.weight, dtype), None)
    proto = proto + mapped.reshape(b, module.num_hyperedges, c)
    z = _linear(x, _mlx_param(module.query_proj, "weight", module.query_proj.weight, dtype), None)
    z = z.reshape(b, n, module.num_heads, module.head_dim).transpose(0, 2, 1, 3)
    proto = proto.reshape(b, module.num_hyperedges, module.num_heads, module.head_dim).transpose(0, 2, 3, 1)
    return mx.softmax(mx.mean((z @ proto) * module.scale, axis=1).transpose(0, 2, 1), axis=-1)


def _hypergraph_convolution(module, x, a, dtype):
    hidden = a @ x
    hidden = _silu(_linear(hidden, _mlx_param(module.W_e, "weight", module.W_e.weight, dtype), None))
    hidden = a.transpose(0, 2, 1) @ hidden
    hidden = _linear(hidden, _mlx_param(module.W_v, "weight", module.W_v.weight, dtype), None)
    return x + _silu(hidden)


def _adaptive_hypergraph_computation(module, x, dtype):
    b, _, h, w = x.shape
    x_flat = x.reshape(b, x.shape[1], h * w).transpose(0, 2, 1)
    a = _adaptive_hyperedge_generation(module.adaptive_hyperedge_gen, x_flat, dtype)
    return _hypergraph_convolution(module.hypergraph_conv, x_flat, a, dtype).transpose(0, 2, 1).reshape(b, -1, h, w)


def _c3ah(module, x, dtype):
    import mlx.core as mx

    return _conv_block(
        module.cv3,
        mx.concatenate(
            (
                _adaptive_hypergraph_computation(module.ahc, _conv_block(module.cv2, x, dtype), dtype),
                _conv_block(module.cv1, x, dtype),
            ),
            axis=1,
        ),
        dtype,
    )


def _hyperace(module, features, dtype):
    import mlx.core as mx

    b2, b3, b4, b5 = features
    size = b4.shape[2:]
    x = _conv_block(
        module.fuse_conv,
        mx.concatenate(
            (
                _resize_bilinear_nchw(b2, size),
                _resize_bilinear_nchw(b3, size),
                b4,
                _resize_bilinear_nchw(b5, size),
            ),
            axis=1,
        ),
        dtype,
    )
    x_h = x[:, : module.c_h]
    x_l = x[:, module.c_h : module.c_h + module.c_l]
    x_s = x[:, module.c_h + module.c_l :]
    high = _conv_block(
        module.high_order_fuse,
        mx.concatenate([_c3ah(branch, x_h, dtype) for branch in module.high_order_branch], axis=1),
        dtype,
    )
    low = _seq(module.low_order_branch, x_l, dtype)
    return _conv_block(module.final_fuse, mx.concatenate((high, low, x_s), axis=1), dtype)


def _gated_fusion(module, f_in, h, dtype):
    return f_in + _mlx_param(module, "gamma", module.gamma, dtype) * h


def _backbone(module, x, dtype):
    x2 = _seq(module.p2, _dsconv_block(module.stem, x, dtype), dtype)
    x3 = _seq(module.p3, x2, dtype)
    x4 = _seq(module.p4, x3, dtype)
    return [x2, x3, x4, _seq(module.p5, x4, dtype)]


def _decoder(module, enc_feats, h_ace, dtype):
    p2, p3, p4, p5 = enc_feats
    d5 = _gated_fusion(
        module.fusion_d5,
        _conv_block(module.skip_p5, p5, dtype),
        _conv_block(module.h_to_d5, _resize_bilinear_nchw(h_ace, p5.shape[2:]), dtype),
        dtype,
    )
    d4 = _ds_c3k2(module.up_d5, _resize_bilinear_nchw(d5, p4.shape[2:]), dtype) + _conv_block(module.skip_p4, p4, dtype)
    d4 = _gated_fusion(
        module.fusion_d4, d4, _conv_block(module.h_to_d4, _resize_bilinear_nchw(h_ace, d4.shape[2:]), dtype), dtype
    )
    d3 = _ds_c3k2(module.up_d4, _resize_bilinear_nchw(d4, p3.shape[2:]), dtype) + _conv_block(module.skip_p3, p3, dtype)
    d3 = _gated_fusion(
        module.fusion_d3, d3, _conv_block(module.h_to_d3, _resize_bilinear_nchw(h_ace, d3.shape[2:]), dtype), dtype
    )
    d2 = _ds_c3k2(module.up_d3, _resize_bilinear_nchw(d3, p2.shape[2:]), dtype) + _conv_block(module.skip_p2, p2, dtype)
    d2 = _gated_fusion(
        module.fusion_d2, d2, _conv_block(module.h_to_d2, _resize_bilinear_nchw(h_ace, d2.shape[2:]), dtype), dtype
    )
    return _ds_c3k2(module.final_d2, d2, dtype)


def _tfc_tdf(module, x, dtype):
    for block in module.blocks:
        shortcut = _conv2d_nchw(block.shortcut, x, dtype)
        x = _segm_module(block.tfc1, x, dtype)
        x = _segm_module(block.tfc2, x + _segm_module(block.tdf, x, dtype), dtype) + shortcut
    return x


def _freq_pixel_shuffle(module, x, dtype):
    x = _dsconv_block(module.conv, x, dtype)
    b, c_r, h, w = x.shape
    out_c = c_r // module.scale
    x = x.reshape(b, out_c, module.scale, h, w).transpose(0, 1, 3, 4, 2).reshape(b, out_c, h, w * module.scale)
    return _tfc_tdf(module.out_conv, x, dtype)


def _progressive_upsample_head(module, x, dtype):
    x = _freq_pixel_shuffle(module.block1, x, dtype)
    x = _freq_pixel_shuffle(module.block2, x, dtype)
    x = _freq_pixel_shuffle(module.block3, x, dtype)
    x = _freq_pixel_shuffle(module.block4, x, dtype)
    if x.shape[-1] != module.target_bins:
        x = _resize_bilinear_nchw(x, (x.shape[2], module.target_bins))
    return _conv2d_nchw(module.final_conv, x, dtype)


def _segm_model(module, x, dtype):
    enc_feats = _backbone(module.backbone, x, dtype)
    dec_feat = _decoder(module.decoder, enc_feats, _hyperace(module.hyperace, enc_feats, dtype), dtype)
    dec_feat = _resize_bilinear_nchw(dec_feat, (x.shape[2], dec_feat.shape[-1]))
    return _progressive_upsample_head(module.upsample_head, dec_feat, dtype)


def _segm_module(module, x, dtype):
    if isinstance(module, torch.nn.Sequential):
        return _seq(module, x, dtype)
    if isinstance(module, hyperace_segm.Conv):
        return _conv_block(module, x, dtype)
    if isinstance(module, hyperace_segm.DSConv):
        return _dsconv_block(module, x, dtype)
    if isinstance(module, hyperace_segm.DS_Bottleneck):
        return _ds_bottleneck(module, x, dtype)
    if isinstance(module, hyperace_segm.DS_C3k):
        return _ds_c3k(module, x, dtype)
    if isinstance(module, hyperace_segm.DS_C3k2):
        return _ds_c3k2(module, x, dtype)
    if isinstance(module, hyperace_segm.TFC_TDF):
        return _tfc_tdf(module, x, dtype)
    if isinstance(module, torch.nn.InstanceNorm2d):
        return _instance_norm2d(module, x, dtype)
    if isinstance(module, torch.nn.SiLU):
        return _silu(x)
    if isinstance(module, torch.nn.Conv2d):
        return _conv2d_nchw(module, x, dtype)
    if isinstance(module, torch.nn.Linear):
        return _linear(
            x,
            _mlx_param(module, "weight", module.weight, dtype),
            None if module.bias is None else _mlx_param(module, "bias", module.bias, dtype),
        )
    if isinstance(module, torch.nn.Identity):
        return x
    raise TypeError(f"unsupported HyperACE SegmModel layer for MLX full backend: {type(module).__name__}")


def _estimate_masks(module, x, dtype):
    import mlx.core as mx

    if isinstance(module, BSRoformerHyperACE) and module.mask_mode != "no_segm":
        masks = []
        segm_input = x.transpose(0, 3, 1, 2)
        for estimator in module.mask_estimators:
            segm = _segm_model(estimator.segm, segm_input, dtype)
            segm = segm.transpose(0, 2, 3, 1).reshape(segm.shape[0], segm.shape[2], -1)
            masks.append(segm if module.mask_mode == "segm_only" else _mask_estimator(estimator, x, dtype) + segm)
        return mx.stack(masks, axis=1)
    return mx.stack([_mask_estimator(estimator, x, dtype) for estimator in module.mask_estimators], axis=1)


def _mask_to_complex_shape(mask):
    return mask.reshape(mask.shape[0], mask.shape[1], mask.shape[2], mask.shape[3] // 2, 2).transpose(0, 1, 3, 2, 4)


def _forward_mask_core(module, stft_repr, dtype):
    b, fs, model_t, complex_dim = stft_repr.shape
    x = stft_repr.transpose(0, 2, 1, 3).reshape(b, model_t, fs * complex_dim)
    x = _band_split(module, x, dtype)

    for time_transformer, freq_transformer in module.layers:
        b, t, f, d = x.shape
        x = _transformer(time_transformer, x.transpose(0, 2, 1, 3).reshape(b * f, t, d), dtype)
        x = x.reshape(b, f, t, d).transpose(0, 2, 1, 3)
        x = _transformer(freq_transformer, x.reshape(b * t, f, d), dtype).reshape(b, t, f, d)

    return _mask_to_complex_shape(_estimate_masks(module, _final_norm(module, x, dtype), dtype))


def _complex_from_ri(x):
    return x[..., 0] + (1j * x[..., 1])


def _ri_from_complex(x):
    import mlx.core as mx

    return mx.stack((x.real, x.imag), axis=-1)


def _mask_stft_repr_bsr(module, stft_repr, dtype):
    mask = _forward_mask_core(module, stft_repr, dtype)
    return _complex_from_ri(stft_repr[:, None]) * _complex_from_ri(mask)


def _mask_stft_repr_mbr(module, stft_repr, context, dtype):
    import mlx.core as mx

    freq_indices = mx.array(module.freq_indices.detach().cpu().numpy())
    x = stft_repr[:, freq_indices]
    masks = _forward_mask_core(module, x, dtype)
    num_stems = len(module.mask_estimators)
    masks_summed = (
        mx.zeros(
            (context["batch"], num_stems, stft_repr.shape[1], stft_repr.shape[-2], 2),
            dtype=masks.dtype,
        )
        .at[:, :, freq_indices, :, :]
        .add(masks)
    )
    denom = mx.array(module.num_bands_per_channel_freq.detach().cpu().numpy(), dtype=masks.dtype)[..., None]
    return _complex_from_ri(stft_repr[:, None]) * _complex_from_ri(masks_summed / mx.maximum(denom, 1e-8))


def mlx_forward_roformer_mx(module, raw_audio, dtype=_COMPUTE_DTYPE):
    if dtype not in (torch.float16, torch.float32):
        raise TypeError("MLX full RoFormer supports torch.float16 or torch.float32 compute dtype")

    mx_dtype = _mlx_dtype(dtype)
    stft_repr, context = _stft_roformer(module, raw_audio.astype(mx_dtype), mx_dtype)
    if isinstance(module, MelBandRoformer):
        masked = _mask_stft_repr_mbr(module, stft_repr, context, dtype)
        length = context["audio_length"] if module.match_input_audio_length else None
    else:
        masked = _mask_stft_repr_bsr(module, stft_repr, dtype)
        length = context["audio_length"]
    return _istft_roformer(module, _ri_from_complex(masked), context, length)


def mlx_forward_roformer(module, raw_audio, dtype=_COMPUTE_DTYPE):
    x_mx = torch_to_mlx_input(raw_audio, dtype=dtype)
    return mlx_to_torch_mps(mlx_forward_roformer_mx(module, x_mx, dtype), raw_audio)
