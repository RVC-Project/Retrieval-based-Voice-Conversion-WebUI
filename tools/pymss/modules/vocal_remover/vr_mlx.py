import torch

from ..bs_roformer.mlx_attention import _linear, _mlx_dtype, _torch_to_mlx_array
from pymss_core.modules.vocal_remover.uvr_lib_v5.vr_network import layers, layers_new, nets, nets_new


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


def _conv_padding(conv):
    padding = conv.padding
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


def _batch_norm2d(module, x, dtype):
    import mlx.core as mx

    if module.training:
        raise TypeError("MLX VR BatchNorm2d supports eval mode only")
    y = x.astype(mx.float32)
    mean = _torch_to_mlx_array(module.running_mean, torch.float32).reshape(1, -1, 1, 1)
    var = _torch_to_mlx_array(module.running_var, torch.float32).reshape(1, -1, 1, 1)
    y = (y - mean) * mx.rsqrt(var + module.eps)
    if module.affine:
        weight = _mlx_param(module, "weight", module.weight, dtype).reshape(1, -1, 1, 1)
        bias = _mlx_param(module, "bias", module.bias, dtype).reshape(1, -1, 1, 1)
        y = y.astype(x.dtype) * weight + bias
    return y.astype(x.dtype)


def _batch_norm1d(module, x, dtype):
    import mlx.core as mx

    if module.training:
        raise TypeError("MLX VR BatchNorm1d supports eval mode only")
    y = x.astype(mx.float32)
    mean = _torch_to_mlx_array(module.running_mean, torch.float32).reshape(1, -1)
    var = _torch_to_mlx_array(module.running_var, torch.float32).reshape(1, -1)
    y = (y - mean) * mx.rsqrt(var + module.eps)
    if module.affine:
        y = y.astype(x.dtype) * _mlx_param(module, "weight", module.weight, dtype).reshape(1, -1)
        y = y + _mlx_param(module, "bias", module.bias, dtype).reshape(1, -1)
    return y.astype(x.dtype)


def _activation(module, x):
    import mlx.core as mx

    if isinstance(module, torch.nn.ReLU):
        return mx.maximum(x, 0)
    if isinstance(module, torch.nn.LeakyReLU):
        return mx.maximum(x, 0) + module.negative_slope * mx.minimum(x, 0)
    if isinstance(module, torch.nn.Sigmoid):
        return mx.sigmoid(x)
    if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Identity)):
        return x
    raise TypeError(f"unsupported VR activation for MLX full backend: {type(module).__name__}")


def _resize_positions_align_corners(in_size, out_size):
    import mlx.core as mx

    if out_size == 1:
        pos = mx.zeros((1,), dtype=mx.float32)
    else:
        pos = mx.arange(out_size, dtype=mx.float32) * ((in_size - 1) / (out_size - 1))
    lower = mx.floor(pos)
    upper = lower + 1
    weight = pos - lower
    lower = mx.clip(lower, 0, in_size - 1).astype(mx.int32)
    upper = mx.clip(upper, 0, in_size - 1).astype(mx.int32)
    return lower, upper, weight


def _resize_bilinear_nchw(x, size=None, scale_factor=None):
    import mlx.core as mx

    if size is None:
        out_h = int(x.shape[2] * scale_factor)
        out_w = int(x.shape[3] * scale_factor)
    else:
        out_h, out_w = int(size[0]), int(size[1])
    in_h, in_w = x.shape[2], x.shape[3]
    if in_h == out_h and in_w == out_w:
        return x
    y0, y1, wy = _resize_positions_align_corners(in_h, out_h)
    x0, x1, wx = _resize_positions_align_corners(in_w, out_w)
    v00 = mx.take(mx.take(x, y0, axis=2), x0, axis=3)
    v01 = mx.take(mx.take(x, y0, axis=2), x1, axis=3)
    v10 = mx.take(mx.take(x, y1, axis=2), x0, axis=3)
    v11 = mx.take(mx.take(x, y1, axis=2), x1, axis=3)
    wy = wy.reshape(1, 1, out_h, 1)
    wx = wx.reshape(1, 1, 1, out_w)
    return v00 * (1 - wy) * (1 - wx) + v01 * (1 - wy) * wx + v10 * wy * (1 - wx) + v11 * wy * wx


def _crop_center(skip, target):
    h, w = target.shape[2], target.shape[3]
    dh = (skip.shape[2] - h) // 2
    dw = (skip.shape[3] - w) // 2
    return skip[:, :, dh : dh + h, dw : dw + w]


def _adaptive_avg_pool_1_none(x):
    import mlx.core as mx

    return mx.mean(x, axis=2, keepdims=True)


def _replicate_pad_freq_bottom(x, pad):
    import mlx.core as mx

    if pad <= 0:
        return x
    last = mx.broadcast_to(x[:, :, -1:, :], (x.shape[0], x.shape[1], pad, x.shape[3]))
    return mx.concatenate((x, last), axis=2)


def _seq(module, x, dtype):
    for child in module:
        x = _module_forward(child, x, dtype)
    return x


def _module_forward(module, x, dtype):
    if isinstance(module, torch.nn.Sequential):
        return _seq(module, x, dtype)
    if isinstance(module, (layers.Conv2DBNActiv, layers.SeperableConv2DBNActiv, layers_new.Conv2DBNActiv)):
        return _seq(module.conv, x, dtype)
    if isinstance(module, torch.nn.Conv2d):
        return _conv2d_nchw(module, x, dtype)
    if isinstance(module, torch.nn.BatchNorm2d):
        return _batch_norm2d(module, x, dtype)
    if isinstance(module, torch.nn.BatchNorm1d):
        return _batch_norm1d(module, x, dtype)
    if isinstance(module, torch.nn.Linear):
        return _linear(
            x,
            _mlx_param(module, "weight", module.weight, dtype),
            None if module.bias is None else _mlx_param(module, "bias", module.bias, dtype),
        )
    if isinstance(module, torch.nn.AdaptiveAvgPool2d):
        if module.output_size != (1, None):
            raise TypeError(f"unsupported VR AdaptiveAvgPool2d output_size: {module.output_size}")
        return _adaptive_avg_pool_1_none(x)
    if isinstance(
        module, (torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.Sigmoid, torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Identity)
    ):
        return _activation(module, x)
    if isinstance(module, layers.ASPPModule):
        return _old_aspp(module, x, dtype)
    if isinstance(module, layers.Decoder):
        return _old_decoder(module, x, None, dtype)
    if isinstance(module, layers_new.ASPPModule):
        return _new_aspp(module, x, dtype)
    if isinstance(module, layers_new.LSTMModule):
        return _new_lstm_module(module, x, dtype)
    if isinstance(module, layers_new.Decoder):
        return _new_decoder(module, x, None, dtype)
    if isinstance(module, nets.BaseASPPNet):
        return _old_base_aspp_net(module, x, dtype)
    if isinstance(module, nets_new.BaseNet):
        return _new_base_net(module, x, dtype)
    raise TypeError(f"unsupported VR layer for MLX full backend: {type(module).__name__}")


def _old_encoder(module, x, dtype):
    skip = _module_forward(module.conv1, x, dtype)
    return _module_forward(module.conv2, skip, dtype), skip


def _old_decoder(module, x, skip, dtype):
    import mlx.core as mx

    x = _resize_bilinear_nchw(x, scale_factor=2)
    if skip is not None:
        x = mx.concatenate((x, _crop_center(skip, x)), axis=1)
    x = _module_forward(module.conv, x, dtype)
    return x if module.dropout is None else x


def _old_aspp(module, x, dtype):
    import mlx.core as mx

    h, w = x.shape[2], x.shape[3]
    features = [
        _resize_bilinear_nchw(_module_forward(module.conv1, x, dtype), size=(h, w)),
        _module_forward(module.conv2, x, dtype),
        _module_forward(module.conv3, x, dtype),
        _module_forward(module.conv4, x, dtype),
        _module_forward(module.conv5, x, dtype),
    ]
    if module.nn_architecture in module.six_layer:
        features.append(_module_forward(module.conv6, x, dtype))
    elif module.nn_architecture in module.seven_layer:
        features.extend((_module_forward(module.conv6, x, dtype), _module_forward(module.conv7, x, dtype)))
    return _module_forward(module.bottleneck, mx.concatenate(features, axis=1), dtype)


def _old_base_aspp_net(module, x, dtype):
    x, skip1 = _old_encoder(module.enc1, x, dtype)
    x, skip2 = _old_encoder(module.enc2, x, dtype)
    x, skip3 = _old_encoder(module.enc3, x, dtype)
    x, skip4 = _old_encoder(module.enc4, x, dtype)

    if module.nn_architecture == 129605:
        x, skip5 = _old_encoder(module.enc5, x, dtype)
        x = _old_decoder(module.dec5, _old_aspp(module.aspp, x, dtype), skip5, dtype)
    else:
        x = _old_aspp(module.aspp, x, dtype)

    x = _old_decoder(module.dec4, x, skip4, dtype)
    x = _old_decoder(module.dec3, x, skip3, dtype)
    x = _old_decoder(module.dec2, x, skip2, dtype)
    return _old_decoder(module.dec1, x, skip1, dtype)


def _old_cascaded_aspp_net(module, x, dtype):
    import mlx.core as mx

    x = x[:, :, : module.max_bin]
    bandwidth = x.shape[2] // 2
    aux1 = mx.concatenate(
        (
            _old_base_aspp_net(module.stg1_low_band_net, x[:, :, :bandwidth], dtype),
            _old_base_aspp_net(module.stg1_high_band_net, x[:, :, bandwidth:], dtype),
        ),
        axis=2,
    )

    hidden = mx.concatenate((x, aux1), axis=1)
    aux2 = _old_base_aspp_net(module.stg2_full_band_net, _module_forward(module.stg2_bridge, hidden, dtype), dtype)

    hidden = mx.concatenate((x, aux1, aux2), axis=1)
    mask = mx.sigmoid(
        _conv2d_nchw(
            module.out,
            _old_base_aspp_net(module.stg3_full_band_net, _module_forward(module.stg3_bridge, hidden, dtype), dtype),
            dtype,
        )
    )
    return _replicate_pad_freq_bottom(mask, module.output_bin - mask.shape[2])


def _new_encoder(module, x, dtype):
    return _module_forward(module.conv2, _module_forward(module.conv1, x, dtype), dtype)


def _new_decoder(module, x, skip, dtype):
    import mlx.core as mx

    x = _resize_bilinear_nchw(x, scale_factor=2)
    if skip is not None:
        x = mx.concatenate((x, _crop_center(skip, x)), axis=1)
    x = _module_forward(module.conv1, x, dtype)
    return x if module.dropout is None else x


def _new_aspp(module, x, dtype):
    import mlx.core as mx

    h, w = x.shape[2], x.shape[3]
    out = mx.concatenate(
        (
            _resize_bilinear_nchw(_module_forward(module.conv1, x, dtype), size=(h, w)),
            _module_forward(module.conv2, x, dtype),
            _module_forward(module.conv3, x, dtype),
            _module_forward(module.conv4, x, dtype),
            _module_forward(module.conv5, x, dtype),
        ),
        axis=1,
    )
    out = _module_forward(module.bottleneck, out, dtype)
    return out if module.dropout is None else out


def _lstm_forward(rnn, x, dtype):
    import mlx.core as mx

    if rnn.num_layers != 1 or rnn.batch_first:
        raise TypeError("MLX VR LSTM supports one-layer non-batch-first LSTMs only")

    def params(suffix):
        return {
            "w_ih": _mlx_param(rnn, f"weight_ih_l0{suffix}", getattr(rnn, f"weight_ih_l0{suffix}"), dtype),
            "w_hh": _mlx_param(rnn, f"weight_hh_l0{suffix}", getattr(rnn, f"weight_hh_l0{suffix}"), dtype),
            "b_ih": _mlx_param(rnn, f"bias_ih_l0{suffix}", getattr(rnn, f"bias_ih_l0{suffix}"), dtype),
            "b_hh": _mlx_param(rnn, f"bias_hh_l0{suffix}", getattr(rnn, f"bias_hh_l0{suffix}"), dtype),
        }

    def run(p, reverse=False):
        steps = range(x.shape[0] - 1, -1, -1) if reverse else range(x.shape[0])
        h = mx.zeros((x.shape[1], rnn.hidden_size), dtype=x.dtype)
        c = mx.zeros_like(h)
        outs = []
        for t in steps:
            gates = _linear(x[t], p["w_ih"], p["b_ih"]) + _linear(h, p["w_hh"], p["b_hh"])
            i, f, g, o = mx.split(gates, 4, axis=-1)
            i, f, o = mx.sigmoid(i), mx.sigmoid(f), mx.sigmoid(o)
            c = f * c + i * mx.tanh(g)
            h = o * mx.tanh(c)
            outs.append(h)
        if reverse:
            outs.reverse()
        return mx.stack(outs, axis=0)

    forward = run(params(""))
    if not rnn.bidirectional:
        return forward
    return mx.concatenate((forward, run(params("_reverse"), reverse=True)), axis=-1)


def _new_lstm_module(module, x, dtype):
    import mlx.core as mx

    batch, _, nbins, nframes = x.shape
    x = _module_forward(module.conv, x, dtype)[:, 0].transpose(2, 0, 1)
    hidden = _lstm_forward(module.lstm, x, dtype)
    hidden = hidden.reshape(-1, hidden.shape[-1])
    hidden = _module_forward(module.dense, hidden, dtype)
    return hidden.reshape(nframes, batch, 1, nbins).transpose(1, 2, 3, 0)


def _new_base_net(module, x, dtype):
    import mlx.core as mx

    enc1 = _module_forward(module.enc1, x, dtype)
    enc2 = _new_encoder(module.enc2, enc1, dtype)
    enc3 = _new_encoder(module.enc3, enc2, dtype)
    enc4 = _new_encoder(module.enc4, enc3, dtype)
    enc5 = _new_encoder(module.enc5, enc4, dtype)

    x = _new_aspp(module.aspp, enc5, dtype)
    x = _new_decoder(module.dec4, x, enc4, dtype)
    x = _new_decoder(module.dec3, x, enc3, dtype)
    x = _new_decoder(module.dec2, x, enc2, dtype)
    x = mx.concatenate((x, _new_lstm_module(module.lstm_dec2, x, dtype)), axis=1)
    return _new_decoder(module.dec1, x, enc1, dtype)


def _new_cascaded_net(module, x, dtype):
    import mlx.core as mx

    x = x[:, :, : module.max_bin]
    bandwidth = x.shape[2] // 2
    low_in = x[:, :, :bandwidth]
    high_in = x[:, :, bandwidth:]

    low1 = _module_forward(module.stg1_low_band_net, low_in, dtype)
    high1 = _module_forward(module.stg1_high_band_net, high_in, dtype)
    aux1 = mx.concatenate((low1, high1), axis=2)

    low2 = _module_forward(module.stg2_low_band_net, mx.concatenate((low_in, low1), axis=1), dtype)
    high2 = _module_forward(module.stg2_high_band_net, mx.concatenate((high_in, high1), axis=1), dtype)
    aux2 = mx.concatenate((low2, high2), axis=2)

    full = _module_forward(module.stg3_full_band_net, mx.concatenate((x, aux1, aux2), axis=1), dtype)
    mask = mx.sigmoid(_conv2d_nchw(module.out, full, dtype))
    return _replicate_pad_freq_bottom(mask, module.output_bin - mask.shape[2])


def mlx_predict_mask_vr_mx(module, x, dtype=torch.float16):
    if dtype not in (torch.float16, torch.float32):
        raise TypeError("MLX full VR supports torch.float16 or torch.float32 compute dtype")
    mx_dtype = _mlx_dtype(dtype)
    x = x.astype(mx_dtype)
    if isinstance(module, nets.CascadedASPPNet):
        mask = _old_cascaded_aspp_net(module, x, dtype)
    elif isinstance(module, nets_new.CascadedNet):
        mask = _new_cascaded_net(module, x, dtype)
    else:
        raise TypeError(f"unsupported VR model for MLX full backend: {type(module).__name__}")
    if module.offset > 0:
        mask = mask[:, :, :, module.offset : -module.offset]
        if mask.shape[3] <= 0:
            raise ValueError("Window size error: h1_shape[3] must be greater than h2_shape[3]")
    return mask
