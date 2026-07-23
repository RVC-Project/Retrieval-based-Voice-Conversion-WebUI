from contextlib import contextmanager, nullcontext

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from typing import Dict

from pymss_core import get_model_from_config as _core_get_model_from_config

from .config import load_config
from .progress import _ProgressContext


def _model_target(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def get_model_from_config(model_type, config_path, model_kwargs_override=None):
    """Instantiate a separation model from a loaded model config.

    Args:
        model_type (Any): Model type value.
        config_path (str | os.PathLike | None): Config path value.
        model_kwargs_override (Any, optional): Model kwargs override value. Defaults to None.

    Returns:
        Any: Computed result."""
    if model_type == "mel_band_roformer":
        model_kwargs_override = dict(model_kwargs_override or {})
        model_kwargs_override.setdefault("zero_dc", False)
        return _core_get_model_from_config(
            model_type, config_path, model_kwargs_override=model_kwargs_override
        )
    if model_type == "bandit_v2":
        config = load_config(config_path)
        from .modules.bandit_v2.bandit import Bandit

        return Bandit(**config.kwargs), config
    return _core_get_model_from_config(model_type, config_path, model_kwargs_override=model_kwargs_override)


def clear_mlx_cache():
    """Clear MLX memory caches when the MLX backend is available.

    Args:
        None: This callable does not accept user-provided arguments.

    Returns:
        None: This callable completes for its side effects."""
    try:
        import mlx.core as mx
    except Exception:
        return

    clear_cache = getattr(mx, "clear_cache", None)
    if clear_cache is None:
        clear_cache = getattr(getattr(mx, "metal", None), "clear_cache", None)
    if clear_cache is not None:
        clear_cache()


def _getWindowingArray(window_size, fade_size):
    """Implement the getWindowingArray helper.

    Args:
        window_size (Any): Window size value.
        fade_size (Any): Fade size value.

    Returns:
        Any: Computed result."""
    if fade_size <= 0:
        return torch.ones(window_size)

    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein
    return window


def _build_chunk_plan(total_length, chunk_size, step, fade_size):
    """Build chunk plan.

    Args:
        total_length (Any): Total length value.
        chunk_size (Any): Chunk size value.
        step (Any): Step value.
        fade_size (Any): Fade size value.

    Returns:
        Any: Built value."""
    starts = list(range(0, total_length, step))
    normal_window = _getWindowingArray(chunk_size, fade_size)

    def window_for(start):
        """Implement the window for helper.

        Args:
            start (Any): Start value.

        Returns:
            Any: Computed result."""
        length = min(chunk_size, total_length - start)
        if start != 0 and start + length < total_length:
            return normal_window
        window = normal_window.clone()
        if start == 0:
            window[:fade_size] = 1
        if start + length >= total_length:
            window[max(0, length - fade_size) : length] = 1
        return window

    return starts, [window_for(start) for start in starts]


def _get_inference_step(config, chunk_size):
    """Return inference step.

    Args:
        config (AttrDict | dict): Loaded pymss configuration.
        chunk_size (Any): Chunk size value.

    Returns:
        Any: Computed result."""
    overlap_size = int(config.inference.get("overlap_size", chunk_size // 2))
    if overlap_size < 0 or overlap_size >= chunk_size:
        raise ValueError("inference.overlap_size must be >= 0 and < audio.chunk_size")
    return chunk_size - overlap_size


def _complete_chunk_count(total_length, chunk_size, step):
    """Implement the complete chunk count helper.

    Args:
        total_length (Any): Total length value.
        chunk_size (Any): Chunk size value.
        step (Any): Step value.

    Returns:
        Any: Computed result."""
    return 0 if total_length < chunk_size else (total_length - chunk_size) // step + 1


def _fold_windows(counter, windows, step, start_offset=0):
    """Implement the fold windows helper.

    Args:
        counter (Any): Counter value.
        windows (Any): Windows value.
        step (Any): Step value.
        start_offset (Any, optional): Start offset value. Defaults to 0.

    Returns:
        None: This callable completes for its side effects."""
    n_chunks = windows.shape[0]
    if n_chunks == 0:
        return

    chunk_size = windows.shape[-1]
    output_length = (n_chunks - 1) * step + chunk_size
    folded_counter = nn.functional.fold(
        windows.transpose(0, 1).unsqueeze(0),
        output_size=(1, output_length),
        kernel_size=(1, chunk_size),
        stride=(1, step),
    )
    counter[..., start_offset : start_offset + output_length] += folded_counter.view(1, 1, output_length)


def _fold_chunk_batch(result, chunks, windows, step, start_offset=0):
    """Implement the fold chunk batch helper.

    Args:
        result (Any): Result value.
        chunks (Any): Chunks value.
        windows (Any): Windows value.
        step (Any): Step value.
        start_offset (Any, optional): Start offset value. Defaults to 0.

    Returns:
        None: This callable completes for its side effects."""
    n_chunks = chunks.shape[0]
    if n_chunks == 0:
        return

    chunk_size = chunks.shape[-1]
    output_length = (n_chunks - 1) * step + chunk_size
    n_sources, n_channels = chunks.shape[1:3]

    folded = nn.functional.fold(
        (chunks * windows[:, None, None, :]).permute(1, 2, 3, 0).reshape(1, n_sources * n_channels * chunk_size, n_chunks),
        output_size=(1, output_length),
        kernel_size=(1, chunk_size),
        stride=(1, step),
    )
    result[..., start_offset : start_offset + output_length] += folded.view(n_sources, n_channels, output_length)


def _ensure_source_dim(x, chunk_batch):
    """Ensure source dim.

    Args:
        x (Any): X value.
        chunk_batch (Any): Chunk batch value.

    Returns:
        Any: Computed result."""
    return x.unsqueeze(1) if x.ndim == chunk_batch.ndim else x


def _fit_tensor_length(x, length):
    """Implement the fit tensor length helper.

    Args:
        x (Any): X value.
        length (Any): Length value.

    Returns:
        Any: Computed result."""
    if x.shape[-1] > length:
        return x[..., :length]
    if x.shape[-1] < length:
        return nn.functional.pad(x, (0, length - x.shape[-1]))
    return x


def _autocast(device, enabled):
    """Implement the autocast helper.

    Args:
        device (Any): Device value.
        enabled (Any): Enabled value.

    Returns:
        Any: Computed result."""
    device_type = torch.device(device).type
    if enabled and device_type in ("cuda", "mps"):
        return torch.amp.autocast(device_type, dtype=torch.float16)
    return nullcontext()


def _inference_context(device):
    if torch.device(device).type == "privateuseone":
        return torch.no_grad()
    return torch.inference_mode()


def _source_names(config):
    """Implement the source names helper.

    Args:
        config (AttrDict | dict): Loaded pymss configuration.

    Returns:
        Any: Computed result."""
    return config.training.instruments if config.training.target_instrument is None else [config.training.target_instrument]


def _normalize_source_indices(config, source_indices):
    """Normalize source indices.

    Args:
        config (AttrDict | dict): Loaded pymss configuration.
        source_indices (Any): Source indices value.

    Returns:
        Any: Computed result."""
    if source_indices is None:
        return None
    source_count = len(_source_names(config))
    indices = tuple(int(index) for index in source_indices)
    if not indices:
        raise ValueError("source_indices must not be empty")
    if len(set(indices)) != len(indices):
        raise ValueError("source_indices must not contain duplicates")
    if min(indices) < 0 or max(indices) >= source_count:
        raise ValueError(f"source_indices must be in range [0, {source_count})")
    return indices


def _source_count(config, source_indices=None):
    """Implement the source count helper.

    Args:
        config (AttrDict | dict): Loaded pymss configuration.
        source_indices (Any, optional): Source indices value. Defaults to None.

    Returns:
        Any: Computed result."""
    return len(_source_names(config)) if source_indices is None else len(source_indices)


def _sources_to_dict(config, estimated_sources, source_indices=None):
    """Implement the sources to dict helper.

    Args:
        config (AttrDict | dict): Loaded pymss configuration.
        estimated_sources (Any): Estimated sources value.
        source_indices (Any, optional): Source indices value. Defaults to None.

    Returns:
        Any: Computed result."""
    names = _source_names(config)
    if source_indices is not None:
        names = [names[index] for index in source_indices]
    return {k: v for k, v in zip(names, estimated_sources)}


def _prepare_mix_for_chunks(mix, border):
    """Implement the prepare mix for chunks helper.

    Args:
        mix (np.ndarray): Mix value.
        border (Any): Border value.

    Returns:
        Any: Computed result."""
    length_init = mix.shape[-1]
    mix = mix.unsqueeze(0) if mix.ndim == 1 else mix
    if length_init > 2 * border and border > 0:
        mix = nn.functional.pad(mix, (border, border), mode="reflect")
    return mix, length_init


def _init_overlap_buffers(config, mix, device, use_fast_path, source_indices=None):
    """Implement the init overlap buffers helper.

    Args:
        config (AttrDict | dict): Loaded pymss configuration.
        mix (np.ndarray): Mix value.
        device (Any): Device value.
        use_fast_path (Any): Use fast path value.
        source_indices (Any, optional): Source indices value. Defaults to None.

    Returns:
        Any: Computed result."""
    req_shape = (_source_count(config, source_indices),) + tuple(mix.shape)
    result_device = device if use_fast_path else "cpu"
    counter_shape = (1, 1, mix.shape[1])
    result = torch.zeros(req_shape, dtype=torch.float32, device=result_device)
    counter = torch.zeros(counter_shape, dtype=torch.float32, device=result_device)
    return result, counter


def _model_mix(mix, device):
    """Implement the model mix helper.

    Args:
        mix (np.ndarray): Mix value.
        device (Any): Device value.

    Returns:
        Any: Computed result."""
    return mix.to(device) if torch.device(device).type != "cpu" else mix


@contextmanager
def _model_source_context(model, source_indices):
    """Implement the model source context helper.

    Args:
        model (str): Model value.
        source_indices (Any): Source indices value.

    Returns:
        None: This callable completes for its side effects."""
    target = _model_target(model)
    sentinel = object()
    previous = getattr(target, "_pymss_source_indices", sentinel)
    if source_indices is not None:
        target._pymss_source_indices = source_indices
    try:
        yield
    finally:
        if previous is sentinel:
            if hasattr(target, "_pymss_source_indices"):
                delattr(target, "_pymss_source_indices")
        else:
            target._pymss_source_indices = previous


def _select_sources(chunks, source_indices, already_selected=False):
    """Implement the select sources helper.

    Args:
        chunks (Any): Chunks value.
        source_indices (Any): Source indices value.
        already_selected (Any, optional): Already selected value. Defaults to False.

    Returns:
        Any: Computed result."""
    if source_indices is None or already_selected:
        return chunks
    index = torch.as_tensor(source_indices, device=chunks.device)
    return chunks.index_select(1, index)


def _run_model_chunk(model, arr, chunk_size, source_indices=None):
    """Run model chunk.

    Args:
        model (str): Model value.
        arr (np.ndarray): Arr value.
        chunk_size (Any): Chunk size value.
        source_indices (Any, optional): Source indices value. Defaults to None.

    Returns:
        Any: Computed result."""
    target = _model_target(model)
    chunks = _fit_tensor_length(_ensure_source_dim(model(arr), arr).float(), chunk_size)
    already_selected = (
        source_indices is not None and hasattr(target, "_active_source_indices") and chunks.shape[1] == len(source_indices)
    )
    return _select_sources(chunks, source_indices, already_selected=already_selected)


def _extract_chunk(mix, start, chunk_size):
    """Implement the extract chunk helper.

    Args:
        mix (np.ndarray): Mix value.
        start (Any): Start value.
        chunk_size (Any): Chunk size value.

    Returns:
        Any: Computed result."""
    length = min(chunk_size, mix.shape[1] - start)
    part = mix[:, start : start + chunk_size]
    if length == chunk_size:
        return part, length
    if length > chunk_size // 2 + 1:
        part = nn.functional.pad(part, (0, chunk_size - length), mode="reflect")
    else:
        part = nn.functional.pad(part, (0, chunk_size - length, 0, 0), mode="constant", value=0)
    return part, length


def _add_weighted_chunk(result, counter, chunk, window, start, length):
    """Implement the add weighted chunk helper.

    Args:
        result (Any): Result value.
        counter (Any): Counter value.
        chunk (Any): Chunk value.
        window (Any): Window value.
        start (Any): Start value.
        length (Any): Length value.

    Returns:
        None: This callable completes for its side effects."""
    device = result.device
    window = window.to(device=device, dtype=torch.float32)[:length]
    result[..., start : start + length] += chunk[..., :length].to(device=device, dtype=torch.float32) * window
    counter[..., start : start + length] += window


def _run_complete_chunks(
    model,
    mix,
    windows,
    result,
    counter,
    chunk_size,
    step,
    batch_size,
    progress,
    source_indices=None,
):
    """Run complete chunks.

    Args:
        model (str): Model value.
        mix (np.ndarray): Mix value.
        windows (Any): Windows value.
        result (Any): Result value.
        counter (Any): Counter value.
        chunk_size (Any): Chunk size value.
        step (Any): Step value.
        batch_size (Any): Batch size value.
        progress (Any): Progress value.
        source_indices (Any, optional): Source indices value. Defaults to None.

    Returns:
        Any: Computed result."""
    n_chunks = _complete_chunk_count(mix.shape[1], chunk_size, step)
    if n_chunks == 0:
        return 0

    n_complete = n_chunks
    if len(windows) > n_chunks:
        n_complete -= n_complete % batch_size
    if n_complete == 0:
        return 0

    inputs = mix.unfold(-1, chunk_size, step).permute(1, 0, 2)[:n_complete]
    fold_windows = torch.stack(windows[:n_complete], dim=0).to(device=result.device, dtype=torch.float32)
    _fold_windows(counter, fold_windows, step)

    for batch_start in range(0, n_complete, batch_size):
        batch_end = min(batch_start + batch_size, n_complete)
        chunks = _run_model_chunk(model, inputs[batch_start:batch_end].contiguous(), chunk_size, source_indices)
        _fold_chunk_batch(
            result,
            chunks,
            fold_windows[batch_start:batch_end],
            step,
            start_offset=batch_start * step,
        )
        progress.update(step * (batch_end - batch_start))

    return n_complete


def _run_tail_chunks(
    model,
    mix,
    starts,
    windows,
    result,
    counter,
    chunk_size,
    step,
    batch_size,
    first_chunk,
    progress,
    source_indices=None,
):
    """Run tail chunks.

    Args:
        model (str): Model value.
        mix (np.ndarray): Mix value.
        starts (Any): Starts value.
        windows (Any): Windows value.
        result (Any): Result value.
        counter (Any): Counter value.
        chunk_size (Any): Chunk size value.
        step (Any): Step value.
        batch_size (Any): Batch size value.
        first_chunk (Any): First chunk value.
        progress (Any): Progress value.
        source_indices (Any, optional): Source indices value. Defaults to None.

    Returns:
        None: This callable completes for its side effects."""
    for batch_start in range(first_chunk, len(starts), batch_size):
        batch_indices = range(batch_start, min(batch_start + batch_size, len(starts)))
        batch = [(_extract_chunk(mix, starts[idx], chunk_size), idx) for idx in batch_indices]
        batch_data = [chunk for (chunk, _), _ in batch]
        chunks = _run_model_chunk(model, torch.stack(batch_data, dim=0), chunk_size, source_indices)
        for j, ((_, length), idx) in enumerate(batch):
            start = starts[idx]
            _add_weighted_chunk(result, counter, chunks[j], windows[idx], start, length)

        progress.update(step * len(batch_data))


def _finalize_overlap(result, counter, length_init, border):
    """Implement the finalize overlap helper.

    Args:
        result (Any): Result value.
        counter (Any): Counter value.
        length_init (Any): Length init value.
        border (Any): Border value.

    Returns:
        Any: Computed result."""
    if length_init > 2 * border and border > 0:
        start, end = border, border + length_init
    else:
        start, end = 0, result.shape[-1]

    result = result[..., start:end]
    counter = counter[..., start:end]
    output_shape = result.shape[:-1] + (end - start,)

    if torch.device(result.device).type != "cuda":
        estimated_sources = (result / counter).cpu().numpy()
        np.nan_to_num(estimated_sources, copy=False, nan=0.0)
        return estimated_sources

    counter_min, counter_max = torch.aminmax(counter)
    divide_counter = bool((counter_min - 1).abs().item() > 1e-6 or (counter_max - 1).abs().item() > 1e-6)
    samples_per_chunk = max(1, (512 * 1024 * 1024) // (max(1, result.shape[0] * result.shape[1]) * 4))
    estimated_sources_t = torch.empty(output_shape, dtype=torch.float32, device="cpu")
    for offset in range(0, result.shape[-1], samples_per_chunk):
        chunk_end = min(offset + samples_per_chunk, result.shape[-1])
        source = result[..., offset:chunk_end]
        if divide_counter:
            source = source / counter[..., offset:chunk_end]
        estimated_sources_t[..., offset:chunk_end].copy_(source)
    estimated_sources = estimated_sources_t.numpy()
    if divide_counter:
        np.nan_to_num(estimated_sources, copy=False, nan=0.0)
    return estimated_sources


def _mlx_reflect_pad_1d(x, left=0, right=0):
    """Implement the mlx reflect pad 1d helper.

    Args:
        x (Any): X value.
        left (Any, optional): Left value. Defaults to 0.
        right (Any, optional): Right value. Defaults to 0.

    Returns:
        Any: Computed result."""
    import mlx.core as mx

    parts = []
    if left > 0:
        parts.append(x[..., 1 : left + 1][..., ::-1])
    parts.append(x)
    if right > 0:
        parts.append(x[..., -right - 1 : -1][..., ::-1])
    return mx.concatenate(parts, axis=-1)


def _mlx_get_windowing_array(window_size, fade_size):
    """Implement the mlx get windowing array helper.

    Args:
        window_size (Any): Window size value.
        fade_size (Any): Fade size value.

    Returns:
        Any: Computed result."""
    import mlx.core as mx

    if fade_size <= 0:
        return mx.ones((window_size,), dtype=mx.float32)
    fadein = mx.linspace(0, 1, fade_size)
    fadeout = mx.linspace(1, 0, fade_size)
    window = mx.ones((window_size,), dtype=mx.float32)
    window = window.at[:fade_size].multiply(fadein)
    window = window.at[-fade_size:].multiply(fadeout)
    return window


def _mlx_build_chunk_plan(total_length, chunk_size, step, fade_size):
    """Implement the mlx build chunk plan helper.

    Args:
        total_length (Any): Total length value.
        chunk_size (Any): Chunk size value.
        step (Any): Step value.
        fade_size (Any): Fade size value.

    Returns:
        Any: Computed result."""
    starts = list(range(0, total_length, step))
    normal_window = _mlx_get_windowing_array(chunk_size, fade_size)
    windows = []
    for start in starts:
        length = min(chunk_size, total_length - start)
        if start != 0 and start + length < total_length:
            windows.append(normal_window)
            continue
        window = normal_window
        if start == 0 and fade_size > 0:
            window = window.at[:fade_size].add(1 - window[:fade_size])
        if start + length >= total_length and fade_size > 0:
            tail = slice(max(0, length - fade_size), length)
            window = window.at[tail].add(1 - window[tail])
        windows.append(window)
    return starts, windows


def _mlx_prepare_mix_for_chunks(mix, border):
    """Implement the mlx prepare mix for chunks helper.

    Args:
        mix (np.ndarray): Mix value.
        border (Any): Border value.

    Returns:
        Any: Computed result."""
    import mlx.core as mx

    length_init = mix.shape[-1]
    mix = mx.array(np.asarray(mix, dtype=np.float32))
    if mix.ndim == 1:
        mix = mix[None, :]
    if length_init > 2 * border and border > 0:
        mix = _mlx_reflect_pad_1d(mix, border, border)
    return mix, length_init


def _mlx_extract_chunk(mix, start, chunk_size):
    """Implement the mlx extract chunk helper.

    Args:
        mix (np.ndarray): Mix value.
        start (Any): Start value.
        chunk_size (Any): Chunk size value.

    Returns:
        Any: Computed result."""
    import mlx.core as mx

    length = min(chunk_size, mix.shape[1] - start)
    part = mix[:, start : start + chunk_size]
    if length == chunk_size:
        return part, length
    pad = chunk_size - length
    if length > chunk_size // 2 + 1:
        part = _mlx_reflect_pad_1d(part, right=pad)
    else:
        part = mx.pad(part, [(0, 0), (0, pad)])
    return part, length


def _mlx_fit_length(x, length):
    """Implement the mlx fit length helper.

    Args:
        x (Any): X value.
        length (Any): Length value.

    Returns:
        Any: Computed result."""
    import mlx.core as mx

    if x.shape[-1] > length:
        return x[..., :length]
    if x.shape[-1] < length:
        return mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, length - x.shape[-1])])
    return x


@contextmanager
def _mlx_clear_cache_after_eval(enabled=False):
    """Clear MLX allocator cache after explicit eval points when requested."""
    if not enabled:
        yield
        return
    import mlx.core as mx

    original_eval = mx.eval

    def eval_and_clear(*args, **kwargs):
        result = original_eval(*args, **kwargs)
        clear_mlx_cache()
        return result

    mx.eval = eval_and_clear
    try:
        yield
    finally:
        mx.eval = original_eval


def _mlx_run_model_chunk(model, arr, chunk_size, clear_cache_after_eval=False):
    """Implement the mlx run model chunk helper.

    Args:
        model (str): Model value.
        arr (np.ndarray): Arr value.
        chunk_size (Any): Chunk size value.

    Returns:
        Any: Computed result."""
    with _mlx_clear_cache_after_eval(clear_cache_after_eval):
        y = model.mlx_forward_mx(arr)
    if y.ndim == arr.ndim:
        y = y[:, None]
    return _mlx_fit_length(y, chunk_size)


def _mlx_select_sources(chunks, source_indices):
    """Implement the mlx select sources helper.

    Args:
        chunks (Any): Chunks value.
        source_indices (Any): Source indices value.

    Returns:
        Any: Computed result."""
    if source_indices is None:
        return chunks

    import mlx.core as mx

    return mx.take(chunks, mx.array(source_indices, dtype=mx.int32), axis=1)


def _mlx_add_weighted_chunk(result, counter, chunk, window, start, length):
    """Implement the mlx add weighted chunk helper.

    Args:
        result (Any): Result value.
        counter (Any): Counter value.
        chunk (Any): Chunk value.
        window (Any): Window value.
        start (Any): Start value.
        length (Any): Length value.

    Returns:
        Any: Computed result."""
    import mlx.core as mx

    window = window[:length].astype(result.dtype)
    weighted = chunk[..., :length].astype(result.dtype) * window
    positions = mx.arange(start, start + length)
    return result.at[:, :, positions].add(weighted), counter.at[:, :, positions].add(window)


def _mlx_finalize_overlap(result, counter, length_init, border):
    """Implement the mlx finalize overlap helper.

    Args:
        result (Any): Result value.
        counter (Any): Counter value.
        length_init (Any): Length init value.
        border (Any): Border value.

    Returns:
        Any: Computed result."""
    import mlx.core as mx

    estimated_sources = result / counter
    if length_init > 2 * border and border > 0:
        estimated_sources = estimated_sources[..., border:-border]
    estimated_sources = np.array(estimated_sources, copy=False)
    np.nan_to_num(estimated_sources, copy=False, nan=0.0)
    return estimated_sources


def _can_demix_mlx_full(model, device):
    """Implement the can demix mlx full helper.

    Args:
        model (str): Model value.
        device (Any): Device value.

    Returns:
        Any: Computed result."""
    return (
        torch.device(device).type == "mps"
        and getattr(model, "mps_model_backend", None) == "mlx_full"
        and hasattr(model, "mps_model_compute_dtype")
        and hasattr(model, "mlx_forward_mx")
    )


def demix_track_mlx_full(config, model, mix, device, pbar=False, source_indices=None, progress_callback=None):
    """Demix a tensor track with the full MLX inference path.

    Args:
        config (AttrDict | dict): Loaded pymss configuration.
        model (str): Model value.
        mix (np.ndarray): Mix value.
        device (Any): Device value.
        pbar (Any, optional): Pbar value. Defaults to False.
        source_indices (Any, optional): Source indices value. Defaults to None.
        progress_callback (Any, optional): Progress callback value. Defaults to None.

    Returns:
        Any: Computed result."""
    import mlx.core as mx

    C = config.audio.chunk_size
    sample_rate = int(config.audio.get("sample_rate", 44100))
    source_indices = _normalize_source_indices(config, source_indices)
    step = _get_inference_step(config, C)
    border = C - step
    fade_size = min(C // 10, border)
    batch_size = config.inference.batch_size

    mix, length_init = _mlx_prepare_mix_for_chunks(mix, border)
    starts, windows = _mlx_build_chunk_plan(mix.shape[1], C, step, fade_size)
    result = mx.zeros((_source_count(config, source_indices), mix.shape[0], mix.shape[1]), dtype=mx.float32)
    counter = mx.zeros((1, 1, mix.shape[1]), dtype=mx.float32)
    progress = _ProgressContext(pbar, mix.shape[1], progress_callback, sample_rate=sample_rate)

    for batch_start in range(0, len(starts), batch_size):
        batch_indices = range(batch_start, min(batch_start + batch_size, len(starts)))
        batch = [(_mlx_extract_chunk(mix, starts[idx], C), idx) for idx in batch_indices]
        batch_count = len(batch)
        chunks = _mlx_run_model_chunk(
            model,
            mx.stack([chunk for (chunk, _), _ in batch], axis=0),
            C,
            clear_cache_after_eval=bool(config.inference.get("mps_mlx_clear_cache", False)),
        )
        chunks = _mlx_select_sources(chunks, source_indices)
        for j, ((_, length), idx) in enumerate(batch):
            result, counter = _mlx_add_weighted_chunk(result, counter, chunks[j], windows[idx], starts[idx], length)
        mx.eval(result, counter)
        del chunks, batch
        clear_mlx_cache()
        progress.update(step * batch_count)

    progress.close()
    progress.emit(mix.shape[1])
    estimated_sources = _mlx_finalize_overlap(result, counter, length_init, border)
    sources = _sources_to_dict(config, estimated_sources, source_indices)
    del result, counter, mix
    clear_mlx_cache()
    return sources


demix_track_mlx_roformer = demix_track_mlx_full


def demix_track(config, model, mix, device, pbar=False, source_indices=None, progress_callback=None):
    """Demix a tensor track with the PyTorch inference path.

    Args:
        config (AttrDict | dict): Loaded pymss configuration.
        model (str): Model value.
        mix (np.ndarray): Mix value.
        device (Any): Device value.
        pbar (Any, optional): Pbar value. Defaults to False.
        source_indices (Any, optional): Source indices value. Defaults to None.
        progress_callback (Any, optional): Progress callback value. Defaults to None.

    Returns:
        Any: Computed result."""
    C = config.audio.chunk_size
    sample_rate = int(config.audio.get("sample_rate", 44100))
    source_indices = _normalize_source_indices(config, source_indices)
    step = _get_inference_step(config, C)
    border = C - step
    fade_size = min(C // 10, border)
    batch_size = config.inference.batch_size

    mix, length_init = _prepare_mix_for_chunks(mix, border)
    chunk_starts, chunk_windows = _build_chunk_plan(mix.shape[1], C, step, fade_size)
    device_type = torch.device(device).type
    use_complete_fast_path = device_type in ("cuda", "cpu")
    mix_device = _model_mix(mix, device)

    with _autocast(device, config.training.get("use_amp", True)):
        with _inference_context(device):
            result, counter = _init_overlap_buffers(config, mix, device, use_complete_fast_path, source_indices)
            progress = _ProgressContext(pbar, mix.shape[1], progress_callback, sample_rate=sample_rate)

            with _model_source_context(model, source_indices):
                complete_chunks = 0
                if use_complete_fast_path:
                    complete_chunks = _run_complete_chunks(
                        model,
                        mix_device,
                        chunk_windows,
                        result,
                        counter,
                        C,
                        step,
                        batch_size,
                        progress,
                        source_indices,
                    )

                _run_tail_chunks(
                    model,
                    mix_device,
                    chunk_starts,
                    chunk_windows,
                    result,
                    counter,
                    C,
                    step,
                    batch_size,
                    complete_chunks,
                    progress,
                    source_indices,
                )
                progress.emit(mix.shape[1])

            progress.close()

            estimated_sources = _finalize_overlap(result, counter, length_init, border)

    return _sources_to_dict(config, estimated_sources, source_indices)


def demix_track_demucs(config, model, mix, device, pbar=False, source_indices=None, progress_callback=None):
    """Demix a tensor track with Demucs-style inference.

    Args:
        config (AttrDict | dict): Loaded pymss configuration.
        model (str): Model value.
        mix (np.ndarray): Mix value.
        device (Any): Device value.
        pbar (Any, optional): Pbar value. Defaults to False.
        source_indices (Any, optional): Source indices value. Defaults to None.
        progress_callback (Any, optional): Progress callback value. Defaults to None.

    Returns:
        Any: Computed result."""
    if _can_demix_mlx_full(model, device):
        return demix_track_mlx_full(
            config,
            model,
            mix.cpu().numpy(),
            device,
            pbar=pbar,
            source_indices=source_indices,
            progress_callback=progress_callback,
        )

    source_indices = _normalize_source_indices(config, source_indices)
    source_names = _source_names(config)
    S = len(source_names)
    sample_rate = int(config.training.samplerate)
    C = sample_rate * config.training.segment
    batch_size = config.inference.batch_size
    step = _get_inference_step(config, C)

    with _autocast(device, config.training.get("use_amp", True)):
        with _inference_context(device):
            req_shape = (_source_count(config, source_indices),) + tuple(mix.shape)
            result = torch.zeros(req_shape, dtype=torch.float32)
            counter = torch.zeros(req_shape, dtype=torch.float32)
            i = 0
            batch_data = []
            batch_locations = []
            progress = _ProgressContext(pbar, mix.shape[1], progress_callback, sample_rate=sample_rate)

            while i < mix.shape[1]:
                part = mix[:, i : i + C].to(device)
                length = part.shape[-1]
                if length < C:
                    part = nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode="constant", value=0)
                batch_data.append(part)
                batch_locations.append((i, length))
                i += step

                if len(batch_data) >= batch_size or (i >= mix.shape[1]):
                    arr = torch.stack(batch_data, dim=0)
                    x = _select_sources(model(arr), source_indices)
                    for j, (start, l) in enumerate(batch_locations):
                        result[..., start : start + l] += x[j][..., :l].cpu()
                        counter[..., start : start + l] += 1.0
                    batch_data, batch_locations = [], []

                progress.emit(min(i, mix.shape[1]))

            progress.close()
            progress.emit(mix.shape[1])

            estimated_sources = (result / counter).cpu().numpy()
            np.nan_to_num(estimated_sources, copy=False, nan=0.0)

    if S == 1 and source_indices is None:
        return estimated_sources
    return _sources_to_dict(config, estimated_sources, source_indices)


def demix(
    config, model, mix: NDArray, device, pbar=False, model_type: str = None, source_indices=None, progress_callback=None
) -> Dict[str, NDArray]:
    """Run chunked model inference and return separated sources.

    Args:
        config (AttrDict | dict): Loaded pymss configuration.
        model (str): Model value.
        mix (np.ndarray): Mix value.
        device (Any): Device value.
        pbar (Any, optional): Pbar value. Defaults to False.
        model_type (Any, optional): Model type value. Defaults to None.
        source_indices (Any, optional): Source indices value. Defaults to None.
        progress_callback (Any, optional): Progress callback value. Defaults to None.

    Returns:
        Any: Computed result."""
    if _can_demix_mlx_full(model, device):
        return demix_track_mlx_full(
            config, model, mix, device, pbar=pbar, source_indices=source_indices, progress_callback=progress_callback
        )
    mix = torch.tensor(mix, dtype=torch.float32)
    if model_type in {"demucs", "tasnet", "legacy_demucs", "legacy_tasnet"}:
        from .modules.legacy_demucs import apply_legacy_model

        sample_rate = int(config.training.samplerate)
        progress = _ProgressContext(
            callback=progress_callback,
            total=mix.shape[1],
            sample_rate=sample_rate,
            message="Processing audio",
        )
        progress.emit(0)
        with _autocast(device, config.training.get("use_amp", True)):
            with _inference_context(device):
                estimates = (
                    apply_legacy_model(
                        model,
                        mix.to(device),
                        shifts=int(config.inference.get("shifts", 0)),
                        split=bool(config.inference.get("split", True)),
                        overlap=float(config.inference.get("overlap", 0.25)),
                        progress=pbar,
                    )
                    .cpu()
                    .numpy()
                )
        progress.emit(mix.shape[1])
        return dict(zip(config.training.instruments, estimates))
    if model_type == "htdemucs":
        return demix_track_demucs(
            config, model, mix, device, pbar=pbar, source_indices=source_indices, progress_callback=progress_callback
        )
    return demix_track(
        config, model, mix, device, pbar=pbar, source_indices=source_indices, progress_callback=progress_callback
    )
