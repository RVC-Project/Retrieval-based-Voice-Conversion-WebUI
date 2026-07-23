import numpy as np
import torch

from ..mlx_utils import mlx_compile_cached


_COMPUTE_DTYPE = torch.float16
_ROTARY_METAL_KERNEL = None
_ROTARY_METAL_UNAVAILABLE = False


def torch_mps_to_mlx(tensor):
    import mlx.core as mx

    return mx.array(tensor.detach().cpu().numpy())


def mlx_to_torch_mps(array, reference):
    return torch.from_numpy(np.array(array, copy=False)).to(device=reference.device, dtype=reference.dtype)


def mlx_bridge_sdpa(q, k, v):
    import mlx.core as mx

    q_mx = torch_mps_to_mlx(q)
    k_mx = torch_mps_to_mlx(k)
    v_mx = torch_mps_to_mlx(v)
    out = mx.fast.scaled_dot_product_attention(q_mx, k_mx, v_mx, scale=q.shape[-1] ** -0.5)
    return mlx_to_torch_mps(out, q)


def _torch_to_mlx_array(tensor, dtype):
    import mlx.core as mx

    return mx.array(tensor.detach().to(dtype=dtype).cpu().numpy())


def _mlx_dtype(dtype):
    import mlx.core as mx

    if dtype == torch.float16:
        return mx.float16
    if dtype == torch.float32:
        return mx.float32
    raise TypeError(f"unsupported MLX bridge dtype: {dtype}")


def _linear(x, weight, bias=None):
    import mlx.core as mx

    y = mx.matmul(x, mx.swapaxes(weight, -1, -2))
    return y if bias is None else y + bias


def _rms_norm(x, gamma):
    import mlx.core as mx

    return x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + 1e-12) * gamma


def _sigmoid(x):
    import mlx.core as mx

    return 1 / (1 + mx.exp(-x))


def _gelu(x):
    import mlx.core as mx

    return 0.5 * x * (1 + mx.erf(x * (2**-0.5)))


def _rotary_metal_kernel():
    global _ROTARY_METAL_KERNEL, _ROTARY_METAL_UNAVAILABLE
    if _ROTARY_METAL_KERNEL is not None:
        return _ROTARY_METAL_KERNEL
    if _ROTARY_METAL_UNAVAILABLE:
        return None

    import mlx.core as mx

    metal_kernel = getattr(getattr(mx, "fast", None), "metal_kernel", None)
    if metal_kernel is None:
        _ROTARY_METAL_UNAVAILABLE = True
        return None

    try:
        _ROTARY_METAL_KERNEL = metal_kernel(
            name="pymss_rotary_qk",
            input_names=["q", "k", "cos", "sin"],
            output_names=["q_out", "k_out"],
            source="""
                uint elem = thread_position_in_grid.x;
                uint d = elem % D;
                uint pair = d >> 1;
                uint pair_base = elem - (d & 1);
                uint seq = (elem / (H * D)) % N;
                uint trig_idx = seq * HALF_D + pair;

                T q_even = q[pair_base];
                T q_odd = q[pair_base + 1];
                T k_even = k[pair_base];
                T k_odd = k[pair_base + 1];
                T c = cos[trig_idx];
                T s = sin[trig_idx];

                if ((d & 1) == 0) {
                    q_out[elem] = q_even * c - q_odd * s;
                    k_out[elem] = k_even * c - k_odd * s;
                } else {
                    q_out[elem] = q_odd * c + q_even * s;
                    k_out[elem] = k_odd * c + k_even * s;
                }
            """,
            ensure_row_contiguous=True,
        )
    except Exception:
        _ROTARY_METAL_UNAVAILABLE = True
        return None
    return _ROTARY_METAL_KERNEL


def _apply_rotary_fallback(q, k, cos, sin):
    import mlx.core as mx

    def rotate(t):
        even, odd = t[..., ::2], t[..., 1::2]
        return mx.stack((even * cos - odd * sin, odd * cos + even * sin), axis=-1).reshape(t.shape)

    return rotate(q), rotate(k)


def _apply_rotary_metal(q, k, cos, sin, dtype):
    kernel = _rotary_metal_kernel()
    if kernel is None or q.ndim != 4 or k.shape != q.shape or q.shape[-1] % 2:
        return None

    _, n, heads, dim = q.shape
    outputs = kernel(
        inputs=[q, k, cos, sin],
        template=[
            ("T", dtype),
            ("N", int(n)),
            ("H", int(heads)),
            ("D", int(dim)),
            ("HALF_D", int(dim // 2)),
        ],
        grid=(q.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[q.shape, k.shape],
        output_dtypes=[q.dtype, k.dtype],
    )
    return outputs[0], outputs[1]


def _rotary_cos_sin(rotary_embed, seq_len, dtype):
    import mlx.core as mx

    cache = getattr(rotary_embed, "_pymss_mlx_cos_sin_cache", None)
    if cache is None:
        rotary_embed._pymss_mlx_cos_sin_cache = cache = {}

    key = (seq_len, dtype, rotary_embed.freqs.data_ptr(), rotary_embed.freqs._version)
    cached = cache.get(key)
    if cached is not None:
        return cached

    freqs = _torch_to_mlx_array(rotary_embed.freqs, torch.float32)
    positions = mx.arange(seq_len, dtype=mx.float32)
    angles = positions[:, None] * freqs[None]
    cos, sin = mx.cos(angles).astype(dtype), mx.sin(angles).astype(dtype)
    cache.clear()
    cache[key] = (cos[None, :, None, :], sin[None, :, None, :])
    return cache[key]


def _apply_rotary(q, k, rotary_embed, dtype):
    cos, sin = _rotary_cos_sin(rotary_embed, q.shape[1], dtype)
    if not getattr(rotary_embed, "_pymss_mlx_disable_metal_rotary", False):
        try:
            out = _apply_rotary_metal(q, k, cos, sin, dtype)
            if out is not None:
                return out
        except Exception as exc:
            rotary_embed._pymss_mlx_disable_metal_rotary = True
            rotary_embed._pymss_mlx_metal_rotary_error = repr(exc)
    return _apply_rotary_fallback(q, k, cos, sin)


def _attention_cache(module, dtype):
    cache = getattr(module, "_pymss_mlx_attention_cache", None)
    params = (
        module.norm.gamma,
        module.to_qkv.weight,
        module.to_qkv.bias,
        module.to_gates.weight,
        module.to_gates.bias,
        module.to_out[0].weight,
        module.to_out[0].bias,
    )
    key = tuple(None if p is None else (p.data_ptr(), p._version, tuple(p.shape), dtype) for p in params)
    if cache is not None and cache.get("key") == key:
        return cache

    cache = {
        "key": key,
        "norm_gamma": _torch_to_mlx_array(module.norm.gamma, dtype),
        "qkv_weight": _torch_to_mlx_array(module.to_qkv.weight, dtype),
        "qkv_bias": None if module.to_qkv.bias is None else _torch_to_mlx_array(module.to_qkv.bias, dtype),
        "gate_weight": _torch_to_mlx_array(module.to_gates.weight, dtype),
        "gate_bias": _torch_to_mlx_array(module.to_gates.bias, dtype),
        "out_weight": _torch_to_mlx_array(module.to_out[0].weight, dtype),
        "out_bias": None if module.to_out[0].bias is None else _torch_to_mlx_array(module.to_out[0].bias, dtype),
    }
    module._pymss_mlx_attention_cache = cache
    return cache


def _mlx_attention(module, x, dtype):
    import mlx.core as mx

    mx_dtype = _mlx_dtype(dtype)
    cache = _attention_cache(module, dtype)
    if module.rotary_embed is None and not getattr(module, "_pymss_mlx_disable_compiled_attention", False):
        try:
            return _mlx_attention_compiled(module, x, dtype, cache)
        except Exception as exc:
            module._pymss_mlx_disable_compiled_attention = True
            module._pymss_mlx_compiled_attention_error = repr(exc)

    x_norm = _rms_norm(x, cache["norm_gamma"])
    qkv = _linear(x_norm, cache["qkv_weight"], cache["qkv_bias"])

    b, n, _ = qkv.shape
    qkv = qkv.reshape(b, n, 3, module.heads, -1)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    if module.rotary_embed is not None:
        q, k = _apply_rotary(q, k, module.rotary_embed, mx_dtype)

    q, k, v = mx.swapaxes(q, 1, 2), mx.swapaxes(k, 1, 2), mx.swapaxes(v, 1, 2)
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5)
    out = mx.swapaxes(out, 1, 2)
    gates = _sigmoid(_linear(x_norm, cache["gate_weight"], cache["gate_bias"]))
    out = (out * gates[..., None]).reshape(b, n, -1)
    return _linear(out, cache["out_weight"], cache["out_bias"])


def _mlx_attention_compiled(module, x, dtype, cache):
    import mlx.core as mx

    if module.rotary_embed is not None:
        raise TypeError("compiled MLX attention path does not include rotary embedding")

    has_qkv_bias = cache["qkv_bias"] is not None
    has_gate_bias = cache["gate_bias"] is not None
    has_out_bias = cache["out_bias"] is not None
    key = (
        tuple(x.shape),
        dtype,
        int(module.heads),
        has_qkv_bias,
        has_gate_bias,
        has_out_bias,
        cache["key"],
    )

    def attention_core(x_arg, norm_gamma, qkv_weight, qkv_bias, gate_weight, gate_bias, out_weight, out_bias):
        x_norm = _rms_norm(x_arg, norm_gamma)
        qkv = _linear(x_norm, qkv_weight, qkv_bias if has_qkv_bias else None)
        b, n, _ = qkv.shape
        qkv = qkv.reshape(b, n, 3, module.heads, -1)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q, k, v = mx.swapaxes(q, 1, 2), mx.swapaxes(k, 1, 2), mx.swapaxes(v, 1, 2)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5)
        out = mx.swapaxes(out, 1, 2)
        gates = _sigmoid(_linear(x_norm, gate_weight, gate_bias if has_gate_bias else None))
        out = (out * gates[..., None]).reshape(b, n, -1)
        return _linear(out, out_weight, out_bias if has_out_bias else None)

    fn = mlx_compile_cached(module, "_pymss_mlx_compiled_attention_cache", key, attention_core)
    dummy = mx.zeros((1,), dtype=_mlx_dtype(dtype))
    return fn(
        x,
        cache["norm_gamma"],
        cache["qkv_weight"],
        cache["qkv_bias"] if has_qkv_bias else dummy,
        cache["gate_weight"],
        cache["gate_bias"] if has_gate_bias else dummy,
        cache["out_weight"],
        cache["out_bias"] if has_out_bias else dummy,
    )


def mlx_bridge_attention(module, x):
    x_mx = torch_mps_to_mlx(x).astype(_mlx_dtype(_COMPUTE_DTYPE))
    out = _mlx_attention(module, x_mx, _COMPUTE_DTYPE)
    return mlx_to_torch_mps(out, x)


def _feed_forward_cache(module, dtype):
    norm, linear_in, activation, _, linear_out, _ = module.net
    if not isinstance(activation, torch.nn.GELU) or activation.approximate != "none":
        raise TypeError("MLX feed-forward bridge only supports torch.nn.GELU(approximate='none')")

    cache = getattr(module, "_pymss_mlx_feed_forward_cache", None)
    params = (
        norm.gamma,
        linear_in.weight,
        linear_in.bias,
        linear_out.weight,
        linear_out.bias,
    )
    key = tuple(None if p is None else (p.data_ptr(), p._version, tuple(p.shape), dtype) for p in params)
    if cache is not None and cache.get("key") == key:
        return cache

    cache = {
        "key": key,
        "norm_gamma": _torch_to_mlx_array(norm.gamma, dtype),
        "linear_in_weight": _torch_to_mlx_array(linear_in.weight, dtype),
        "linear_in_bias": None if linear_in.bias is None else _torch_to_mlx_array(linear_in.bias, dtype),
        "linear_out_weight": _torch_to_mlx_array(linear_out.weight, dtype),
        "linear_out_bias": None if linear_out.bias is None else _torch_to_mlx_array(linear_out.bias, dtype),
    }
    module._pymss_mlx_feed_forward_cache = cache
    return cache


def _mlx_feed_forward(module, x, dtype):
    cache = _feed_forward_cache(module, dtype)
    if not getattr(module, "_pymss_mlx_disable_compiled_ffn", False):
        try:
            return _mlx_feed_forward_compiled(module, x, dtype, cache)
        except Exception as exc:
            module._pymss_mlx_disable_compiled_ffn = True
            module._pymss_mlx_compiled_ffn_error = repr(exc)

    return _mlx_feed_forward_fallback(x, cache)


def _mlx_feed_forward_fallback(x, cache):
    x = _rms_norm(x, cache["norm_gamma"])
    x = _linear(x, cache["linear_in_weight"], cache["linear_in_bias"])
    x = _gelu(x)
    return _linear(x, cache["linear_out_weight"], cache["linear_out_bias"])


def _mlx_feed_forward_compiled(module, x, dtype, cache):
    import mlx.core as mx

    compiled_cache = getattr(module, "_pymss_mlx_compiled_feed_forward_cache", None)
    if compiled_cache is None:
        module._pymss_mlx_compiled_feed_forward_cache = compiled_cache = {}

    has_in_bias = cache["linear_in_bias"] is not None
    has_out_bias = cache["linear_out_bias"] is not None
    key = (tuple(x.shape), dtype, has_in_bias, has_out_bias, cache["key"])
    fn = compiled_cache.get(key)
    if fn is None:

        def ffn_core(x_arg, norm_gamma, linear_in_weight, linear_in_bias, linear_out_weight, linear_out_bias):
            x_arg = _rms_norm(x_arg, norm_gamma)
            x_arg = _linear(x_arg, linear_in_weight, linear_in_bias if has_in_bias else None)
            x_arg = _gelu(x_arg)
            return _linear(x_arg, linear_out_weight, linear_out_bias if has_out_bias else None)

        fn = mx.compile(ffn_core)
        compiled_cache[key] = fn

    dummy = mx.zeros((1,), dtype=_mlx_dtype(dtype))
    return fn(
        x,
        cache["norm_gamma"],
        cache["linear_in_weight"],
        cache["linear_in_bias"] if has_in_bias else dummy,
        cache["linear_out_weight"],
        cache["linear_out_bias"] if has_out_bias else dummy,
    )


def _norm_gamma_cache(module, dtype):
    if isinstance(module, torch.nn.Identity):
        return None
    if not hasattr(module, "gamma"):
        raise TypeError("MLX transformer bridge only supports RMSNorm or Identity output norm")

    cache = getattr(module, "_pymss_mlx_norm_cache", None)
    key = (module.gamma.data_ptr(), module.gamma._version, tuple(module.gamma.shape), dtype)
    if cache is not None and cache.get("key") == key:
        return cache["gamma"]

    cache = {"key": key, "gamma": _torch_to_mlx_array(module.gamma, dtype)}
    module._pymss_mlx_norm_cache = cache
    return cache["gamma"]


def _mlx_output_norm(module, x, dtype):
    gamma = _norm_gamma_cache(module, dtype)
    return x if gamma is None else _rms_norm(x, gamma)


def mlx_bridge_transformer(module, x):
    x_mx = torch_mps_to_mlx(x).astype(_mlx_dtype(_COMPUTE_DTYPE))
    for attn, ff in module.layers:
        x_mx = _mlx_attention(attn, x_mx, _COMPUTE_DTYPE) + x_mx
        x_mx = _mlx_feed_forward(ff, x_mx, _COMPUTE_DTYPE) + x_mx
    x_mx = _mlx_output_norm(module.norm, x_mx, _COMPUTE_DTYPE)
    return mlx_to_torch_mps(x_mx, x)
