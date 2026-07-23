def mlx_periodic_hann_window(length, dtype):
    import mlx.core as mx

    length = int(length)
    if length <= 0:
        return mx.zeros((0,), dtype=dtype)
    if length == 1:
        return mx.ones((1,), dtype=dtype)
    window = mx.hanning(length + 1)[:-1]
    return window.astype(dtype)


def mlx_compile_cached(module, cache_name, key, fn):
    import mlx.core as mx

    cache = getattr(module, cache_name, None)
    if cache is None:
        cache = {}
        setattr(module, cache_name, cache)
    compiled = cache.get(key)
    if compiled is None:
        compiled = mx.compile(fn)
        cache[key] = compiled
    return compiled
