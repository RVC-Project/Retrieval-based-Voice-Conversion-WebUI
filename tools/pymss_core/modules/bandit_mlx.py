import numpy as np
import torch

from .mlx_utils import mlx_periodic_hann_window
from .bandit.tfmodel import ResidualRNN, Transpose
from .bs_roformer.mlx_attention import _gelu, _linear, _mlx_dtype, _torch_to_mlx_array, mlx_to_torch_mps


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


def _pad_last(x, left, right, mode="constant"):
    import mlx.core as mx

    if left <= 0 and right <= 0:
        return x
    if mode == "constant":
        return mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(left, right)])
    if mode != "reflect":
        raise TypeError(f"MLX Bandit STFT does not support pad_mode={mode!r}")
    if x.shape[-1] <= max(left, right):
        raise ValueError("reflect padding requires input length greater than padding")
    parts = []
    if left > 0:
        parts.append(x[..., 1 : left + 1][..., ::-1])
    parts.append(x)
    if right > 0:
        parts.append(x[..., -right - 1 : -1][..., ::-1])
    return mx.concatenate(parts, axis=-1)


def _spectral_stft(stft_module, raw_audio, dtype):
    import mlx.core as mx

    leading_shape = raw_audio.shape[:-1]
    length = raw_audio.shape[-1]
    n_fft = int(stft_module.n_fft)
    win_length = int(stft_module.win_length)
    hop = int(stft_module.hop_length)
    x = raw_audio.reshape(-1, length).astype(dtype)
    if stft_module.center:
        x = _pad_last(x, n_fft // 2, n_fft // 2, stft_module.pad_mode)
    frames = 1 + (x.shape[-1] - n_fft) // hop
    framed = mx.as_strided(x, shape=(x.shape[0], frames, n_fft), strides=(x.shape[-1], hop, 1))
    window = _hann_window(win_length, dtype)
    if win_length < n_fft:
        left = (n_fft - win_length) // 2
        window = mx.pad(window, [(left, n_fft - win_length - left)])
    elif win_length > n_fft:
        raise ValueError("MLX Bandit STFT does not support win_length > n_fft")
    spec = mx.fft.rfft(framed * window, n=n_fft, axis=-1)
    if stft_module.normalized:
        spec = spec / np.sqrt(n_fft)
    spec = mx.moveaxis(spec, -1, -2)
    return spec.reshape(*leading_shape, spec.shape[-2], spec.shape[-1]), {
        "n_fft": n_fft,
        "win_length": win_length,
        "hop": hop,
        "window": window,
        "normalized": stft_module.normalized,
        "center": stft_module.center,
        "dtype": dtype,
    }


def _spectral_istft(istft_module, spec, context, length):
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
    elif length is not None:
        audio = audio[..., :length]
    return audio.reshape(*leading_shape, audio.shape[-1])


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


def _group_norm_nchw(module, x, dtype):
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


def _activation(module, x):
    import mlx.core as mx

    if isinstance(module, torch.nn.Tanh):
        return mx.tanh(x)
    if isinstance(module, torch.nn.ReLU):
        return mx.maximum(x, 0)
    if isinstance(module, torch.nn.GELU):
        return _gelu(x)
    if isinstance(module, torch.nn.ELU):
        return mx.where(x > 0, x, module.alpha * (mx.exp(x) - 1))
    if isinstance(module, torch.nn.Identity):
        return x
    raise TypeError(f"unsupported Bandit activation for MLX full backend: {type(module).__name__}")


def _glu(x, axis=-1):
    import mlx.core as mx

    a, b = mx.split(x, 2, axis=axis)
    return a * mx.sigmoid(b)


def _norm_fc(module, xb, dtype):
    if hasattr(module, "combined"):
        xb = _layer_norm(module.combined[0], xb, dtype)
        return _linear(
            xb,
            _mlx_param(module.combined[1], "weight", module.combined[1].weight, dtype),
            _mlx_param(module.combined[1], "bias", module.combined[1].bias, dtype),
        )

    batch, n_time, in_channels, ribw = xb.shape
    xb = _layer_norm(module.norm, xb.reshape(batch, n_time, in_channels * ribw), dtype)
    if module.treat_channel_as_feature:
        return _linear(
            xb, _mlx_param(module.fc, "weight", module.fc.weight, dtype), _mlx_param(module.fc, "bias", module.fc.bias, dtype)
        )
    out = _linear(
        xb.reshape(batch, n_time, in_channels, ribw),
        _mlx_param(module.fc, "weight", module.fc.weight, dtype),
        _mlx_param(module.fc, "bias", module.fc.bias, dtype),
    )
    return out.reshape(batch, n_time, -1)


def _band_split(module, x, dtype):
    import mlx.core as mx

    batch, in_channels, _, n_time = x.shape
    xr = mx.stack((x.real, x.imag), axis=-1)
    if module.complex_order == "reim_freq":
        xr = xr.transpose(0, 3, 1, 4, 2)
    elif module.complex_order == "freq_reim":
        xr = xr.transpose(0, 3, 1, 2, 4)
    else:
        raise ValueError(f"unsupported complex_order: {module.complex_order}")

    outs = []
    for i, nfm in enumerate(module.norm_fc_modules):
        fstart, fend = module.band_specs[i]
        if module.complex_order == "reim_freq":
            xb = xr[..., fstart:fend].reshape(batch, n_time, in_channels, -1)
        else:
            xb = xr[:, :, :, fstart:fend].reshape(batch, n_time, -1)
        outs.append(_norm_fc(nfm, xb.reshape(batch, n_time, -1) if module.flatten_input else xb, dtype))
    return mx.stack(outs, axis=1)


def _rnn_forward(rnn, x, dtype):
    import mlx.core as mx

    def params(suffix):
        return {
            "w_ih": _mlx_param(rnn, f"weight_ih_l0{suffix}", getattr(rnn, f"weight_ih_l0{suffix}"), dtype),
            "w_hh": _mlx_param(rnn, f"weight_hh_l0{suffix}", getattr(rnn, f"weight_hh_l0{suffix}"), dtype),
            "b_ih": _mlx_param(rnn, f"bias_ih_l0{suffix}", getattr(rnn, f"bias_ih_l0{suffix}"), dtype) if rnn.bias else None,
            "b_hh": _mlx_param(rnn, f"bias_hh_l0{suffix}", getattr(rnn, f"bias_hh_l0{suffix}"), dtype) if rnn.bias else None,
        }

    def affine(inp, h, p):
        gates = _linear(inp, p["w_ih"], p["b_ih"]) + _linear(h, p["w_hh"], p["b_hh"])
        return gates

    def run_gru(inp, p, reverse=False):
        steps = range(inp.shape[1] - 1, -1, -1) if reverse else range(inp.shape[1])
        h = mx.zeros((inp.shape[0], rnn.hidden_size), dtype=inp.dtype)
        outs = []
        for t in steps:
            gi = _linear(inp[:, t], p["w_ih"], p["b_ih"])
            gh = _linear(h, p["w_hh"], p["b_hh"])
            i_r, i_z, i_n = mx.split(gi, 3, axis=-1)
            h_r, h_z, h_n = mx.split(gh, 3, axis=-1)
            reset = mx.sigmoid(i_r + h_r)
            update = mx.sigmoid(i_z + h_z)
            new = mx.tanh(i_n + reset * h_n)
            h = (1 - update) * new + update * h
            outs.append(h)
        if reverse:
            outs.reverse()
        return mx.stack(outs, axis=1)

    def run_lstm(inp, p, reverse=False):
        steps = range(inp.shape[1] - 1, -1, -1) if reverse else range(inp.shape[1])
        h = mx.zeros((inp.shape[0], rnn.hidden_size), dtype=inp.dtype)
        c = mx.zeros_like(h)
        outs = []
        for t in steps:
            i, f, g, o = mx.split(affine(inp[:, t], h, p), 4, axis=-1)
            i, f, o = mx.sigmoid(i), mx.sigmoid(f), mx.sigmoid(o)
            c = f * c + i * mx.tanh(g)
            h = o * mx.tanh(c)
            outs.append(h)
        if reverse:
            outs.reverse()
        return mx.stack(outs, axis=1)

    if rnn.num_layers != 1 or not rnn.batch_first:
        raise TypeError("MLX Bandit RNN supports one-layer batch_first RNNs only")
    if isinstance(rnn, torch.nn.GRU):
        forward = run_gru(x, params(""))
        if not rnn.bidirectional:
            return forward
        return mx.concatenate((forward, run_gru(x, params("_reverse"), reverse=True)), axis=-1)
    if isinstance(rnn, torch.nn.LSTM):
        forward = run_lstm(x, params(""))
        if not rnn.bidirectional:
            return forward
        return mx.concatenate((forward, run_lstm(x, params("_reverse"), reverse=True)), axis=-1)
    raise TypeError(f"unsupported Bandit RNN for MLX full backend: {type(rnn).__name__}")


def _residual_rnn(module, z, dtype):
    z0 = z
    if module.use_layer_norm:
        z = _layer_norm(module.norm, z, dtype)
    else:
        z = _group_norm_nchw(module.norm, z.transpose(0, 3, 1, 2), dtype).transpose(0, 2, 3, 1)

    batch, n_uncrossed, n_across, emb_dim = z.shape
    if module.use_batch_trick:
        z = _rnn_forward(module.rnn, z.reshape(batch * n_uncrossed, n_across, emb_dim), dtype)
        z = z.reshape(batch, n_uncrossed, n_across, -1)
    else:
        import mlx.core as mx

        z = mx.stack([_rnn_forward(module.rnn, z[:, i], dtype) for i in range(n_uncrossed)], axis=1)
    return (
        _linear(
            z, _mlx_param(module.fc, "weight", module.fc.weight, dtype), _mlx_param(module.fc, "bias", module.fc.bias, dtype)
        )
        + z0
    )


def _tf_model(module, z, dtype):
    if module.parallel_mode:
        for sbm_t, sbm_f in module.seqband:
            zt = _residual_rnn(sbm_t, z, dtype)
            zf = _residual_rnn(sbm_f, z.transpose(0, 2, 1, 3), dtype)
            z = zt + zf.transpose(0, 2, 1, 3)
        return z

    if isinstance(module.seqband, torch.nn.Sequential):
        for layer in module.seqband:
            if isinstance(layer, ResidualRNN):
                z = _residual_rnn(layer, z, dtype)
            elif isinstance(layer, Transpose):
                z = z.swapaxes(layer.dim0, layer.dim1)
            else:
                raise TypeError(f"unsupported Bandit TF layer for MLX full backend: {type(layer).__name__}")
        return z

    for sbm in module.seqband:
        z = _residual_rnn(sbm, z, dtype)
        z = z.swapaxes(1, 2)
    return z


def _norm_mlp(module, qb, dtype):
    x = _layer_norm(module.norm, qb, dtype)
    x = _linear(
        x,
        _mlx_param(module.hidden[0], "weight", module.hidden[0].weight, dtype),
        _mlx_param(module.hidden[0], "bias", module.hidden[0].bias, dtype),
    )
    x = _activation(module.hidden[1], x)
    output = module.output[0]
    x = _linear(x, _mlx_param(output, "weight", output.weight, dtype), _mlx_param(output, "bias", output.bias, dtype))
    x = _glu(x, axis=-1)

    batch, n_time, _ = x.shape
    if module.complex_mask:
        x = x.reshape(batch, n_time, module.in_channels, module.bandwidth, 2)
        x = x[..., 0] + (1j * x[..., 1])
    else:
        x = x.reshape(batch, n_time, module.in_channels, module.bandwidth)
    return x.transpose(0, 2, 3, 1)


def _append_cond(module, q, cond):
    import mlx.core as mx

    if cond is not None:
        batch, n_bands, n_time, _ = q.shape
        if cond.ndim == 2:
            cond = mx.broadcast_to(cond[:, None, None, :], (batch, n_bands, n_time, cond.shape[-1]))
        elif cond.ndim != 3:
            raise ValueError(f"Invalid cond shape: {cond.shape}")
        return mx.concatenate((q, cond), axis=-1)
    if module.cond_dim <= 0:
        return q
    batch, n_bands, n_time, _ = q.shape
    return mx.concatenate((q, mx.ones((batch, n_bands, n_time, module.cond_dim), dtype=q.dtype)), axis=-1)


def _mask_estimator(module, q, dtype, cond=None):
    import mlx.core as mx

    q = _append_cond(module, q, cond)
    if getattr(module, "n_freq", 0) <= 0:
        return mx.concatenate([_norm_mlp(nmlp, q[:, b], dtype) for b, nmlp in enumerate(module.norm_mlp)], axis=2)

    batch, _, n_time, _ = q.shape
    mask_real = mx.zeros((batch, module.in_channels, module.n_freq, n_time), dtype=mx.float32)
    mask_imag = mx.zeros_like(mask_real)
    for band_index, nmlp in enumerate(module.norm_mlp):
        fstart, fend = module.band_specs[band_index]
        mask = _norm_mlp(nmlp, q[:, band_index], dtype)
        if module.use_freq_weights:
            fw = _torch_to_mlx_array(module.get_buffer(f"freq_weights/{band_index}"), dtype)
            mask = mask * fw.reshape(1, 1, -1, 1)
        padding = [(0, 0), (0, 0), (fstart, module.n_freq - fend), (0, 0)]
        mask_real = mask_real + mx.pad(mask.real.astype(mask_real.dtype), padding)
        mask_imag = mask_imag + mx.pad(mask.imag.astype(mask_imag.dtype), padding)
    return mask_real + (1j * mask_imag)


def _bsrnn_core(module, x, dtype):
    batch, in_chan, n_freq, n_time = x.shape
    x = x.reshape(-1, 1, n_freq, n_time)
    q = _tf_model(module.tf_model, _band_split(module.band_split, x, dtype), dtype)
    return [_mask_estimator(mask_estimator, q, dtype) * x for mask_estimator in module.mask_estim.values()]


def mlx_forward_bandit_mx(module, raw_audio, dtype=torch.float16):
    import mlx.core as mx

    if dtype not in (torch.float16, torch.float32):
        raise TypeError("MLX full Bandit supports torch.float16 or torch.float32 compute dtype")
    mx_dtype = _mlx_dtype(dtype)
    init_shape = raw_audio.shape
    mono = raw_audio.reshape(-1, 1, raw_audio.shape[-1]).astype(mx_dtype)
    x, context = _spectral_stft(module.stft, mono, mx_dtype)
    length = mono.shape[-1]

    if hasattr(module, "bsrnn"):
        specs = _bsrnn_core(module.bsrnn, x, dtype)
        stems = module.stems
    else:
        q = _tf_model(module.tf_model, _band_split(module.band_split, x, dtype), dtype)
        specs = [_mask_estimator(mask_estimator, q, dtype) * x for mask_estimator in module.mask_estim.values()]
        stems = module.stems

    estimates = [_spectral_istft(module.istft, spec, context, length) for spec in specs]
    estimates = [estimate.reshape(-1, init_shape[1], init_shape[2]) for estimate in estimates]
    return mx.stack(estimates, axis=1)


def mlx_forward_bandit(module, raw_audio, dtype=torch.float16):
    x_mx = torch_to_mlx_input(raw_audio, dtype=dtype)
    return mlx_to_torch_mps(mlx_forward_bandit_mx(module, x_mx, dtype), raw_audio)
