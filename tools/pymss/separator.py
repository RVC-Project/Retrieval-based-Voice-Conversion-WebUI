import gc
import os
import logging
import re
from contextlib import contextmanager, nullcontext
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import torch
import numpy as np
import platform
import subprocess
from time import time
from tqdm import tqdm

from .audio_io import load_audio, save_audio
from .utils import clear_mlx_cache, demix, get_model_from_config
from .logger import get_separation_logger, set_log_level
from .config import AttrDict


INFERENCE_PARAM_TARGETS = {
    "batch_size": "inference",
    "overlap_size": "inference",
    "chunk_size": "audio",
    "standardize": "inference",  # legacy input standardization, will be mapped to inference.normalize
    "normalize": "inference",  # output peak normalization, takes precedence over standardize if both are present
    "mask_mode": "inference",
    "window_size": "inference",
    "aggression": "inference",
    "enable_tta": "inference",
    "enable_post_process": "inference",
    "post_process_threshold": "inference",
    "high_end_process": "inference",
    "use_amp": "inference",
    "cuda_attention_backend": "inference",
    "mps_attention_backend": "inference",
    "mps_mlx_min_tokens": "inference",
    "mps_mlx_clear_cache": "inference",
    "mps_model_backend": "inference",
    "mps_model_compute_dtype": "inference",
    "model_dtype": "inference",
    "fuse_conv_bn": "inference",
    "use_channels_last": "inference",
    "shifts": "inference",
    "split": "inference",
    "overlap": "inference",
    "stem_batch_size": "inference",
}
PASSTHROUGH_INFERENCE_PARAMS = frozenset(
    {
        "standardize",
        "normalize",
        "mask_mode",
        "enable_tta",
        "enable_post_process",
        "high_end_process",
        "use_amp",
        "cuda_attention_backend",
        "mps_attention_backend",
        "mps_mlx_clear_cache",
        "mps_model_backend",
        "mps_model_compute_dtype",
        "model_dtype",
        "fuse_conv_bn",
        "use_channels_last",
        "split",
    }
)
FAST_INIT_MODEL_TYPES = {"bs_roformer", "bs_roformer_hyperace", "mel_band_roformer"}
DML_FP16_MODEL_TYPES = frozenset(FAST_INIT_MODEL_TYPES)
LEGACY_DEMUCS_MODEL_TYPES = {"demucs", "tasnet", "legacy_demucs", "legacy_tasnet"}
OUTPUT_NORMALIZE_TARGET_DBFS = -0.01
OUTPUT_NORMALIZE_PEAK = 10 ** (OUTPUT_NORMALIZE_TARGET_DBFS / 20)
_DML_CACHE_ATTRIBUTE_NAMES = frozenset(
    {
        "cache",
        "_group_cache",
        "_layer_group_cache",
        "_index_cache",
        "_packed_layer_group_cache",
        "_stft_window_cache",
        "_pymss_cos_sin_cache",
        "_gamma_dtype_cache",
    }
)


def _device_type(device):
    """Return a Torch device type without failing during teardown."""
    try:
        return torch.device(device).type
    except (RuntimeError, TypeError):
        return str(device).split(":", 1)[0].lower()


def _clear_cache_container(value, seen):
    """Clear nested mutable cache containers, dropping cached tensor references."""
    value_id = id(value)
    if value_id in seen:
        return
    seen.add(value_id)

    if isinstance(value, dict):
        for cached_value in tuple(value.values()):
            _clear_cache_container(cached_value, seen)
        value.clear()
    elif isinstance(value, (list, set, deque)):
        for cached_value in tuple(value):
            _clear_cache_container(cached_value, seen)
        value.clear()
    elif isinstance(value, tuple):
        for cached_value in value:
            _clear_cache_container(cached_value, seen)


def _clear_dml_model_caches(model, logger=None):
    """Drop DML tensor caches held outside registered module buffers."""
    if model is None:
        return 0

    candidates = [model]
    model_run = getattr(model, "model_run", None)
    if model_run is not None and model_run is not model:
        candidates.append(model_run)

    cleared = 0
    seen_modules = set()
    for candidate in candidates:
        modules = getattr(candidate, "modules", None)
        if not callable(modules):
            continue
        try:
            module_items = tuple(modules())
        except Exception as exc:
            if logger is not None:
                logger.debug(f"Could not enumerate DirectML model caches during close: {exc}")
            continue

        for module in module_items:
            module_id = id(module)
            if module_id in seen_modules:
                continue
            seen_modules.add(module_id)
            try:
                attributes = tuple(vars(module).items())
            except TypeError:
                continue

            for name, value in attributes:
                is_cache = name in _DML_CACHE_ATTRIBUTE_NAMES or name.endswith("_cache")
                if not is_cache or not isinstance(value, (dict, list, set, deque)):
                    continue
                try:
                    _clear_cache_container(value, set())
                    cleared += 1
                except Exception as exc:
                    if logger is not None:
                        logger.debug(f"Could not clear DirectML cache {name}: {exc}")

    return cleared


def _resolve_public_device(device, inference_params, logger):
    """Resolve public device.

    Args:
        device (Any): Device value.
        inference_params (dict | None): Inference params value.
        logger (logging.Logger | None): Optional logger for progress messages.

    Returns:
        Any: Resolved value."""
    inference_params = dict(inference_params or {})
    requested_device = device
    if requested_device == "mlx":
        if not torch.backends.mps.is_available():
            raise RuntimeError("device='mlx' requires Apple Silicon MPS support")
        inference_params.setdefault("mps_model_backend", "mlx_full")
        inference_params.setdefault("mps_model_compute_dtype", "float16")
        inference_params.setdefault("mps_mlx_clear_cache", True)
        logger.debug("Mapping device='mlx' to device='mps' with MLX full model backend")
        return "mps", inference_params
    requested_type = None
    try:
        requested_type = torch.device(requested_device).type
    except (RuntimeError, TypeError):
        pass
    if requested_device not in {"auto", "cpu", "cuda", "mps"} and requested_type != "privateuseone":
        raise ValueError("device must be 'auto', 'cpu', 'cuda', 'mps', 'mlx', or a DirectML device")
    return requested_device, inference_params


def _select_device(device, device_ids, logger):
    """Implement the select device helper.

    Args:
        device (Any): Device value.
        device_ids (Any): Device ids value.
        logger (logging.Logger | None): Optional logger for progress messages.

    Returns:
        Any: Computed result."""
    try:
        device_type = torch.device(device).type
    except (RuntimeError, TypeError):
        device_type = None
    if device_type == "privateuseone":
        logger.debug("DirectML device selected: %s", device)
        return str(device)
    if device not in ["cpu", "cuda", "mps"]:
        if torch.cuda.is_available():
            logger.debug("CUDA is available in Torch, setting Torch device to CUDA")
            return f"cuda:{device_ids[0]}"
        if torch.backends.mps.is_available():
            logger.debug("Apple Silicon MPS/CoreML is available in Torch, setting Torch device to MPS")
            return "mps"
        return "cpu"

    if device == "cpu":
        logger.warning("No hardware acceleration could be configured, running in CPU mode")
    return device


def _prefer_mlx_for_auto(requested_device, selected_device, inference_params, logger):
    """Implement the prefer mlx for auto helper.

    Args:
        requested_device (Any): Requested device value.
        selected_device (Any): Selected device value.
        inference_params (dict | None): Inference params value.
        logger (logging.Logger | None): Optional logger for progress messages.

    Returns:
        Any: Computed result."""
    if requested_device == "auto" and torch.device(selected_device).type == "mps":
        if "mps_model_backend" not in inference_params:
            inference_params["mps_model_backend"] = "mlx_full"
            inference_params.setdefault("mps_model_compute_dtype", "float16")
            inference_params.setdefault("mps_mlx_clear_cache", True)
            logger.debug("Auto device selected MPS, enabling MLX full model backend")
        elif inference_params.get("mps_model_backend") == "mlx_full":
            inference_params.setdefault("mps_mlx_clear_cache", True)
    return inference_params


def _unwrap_state_dict(state_dict):
    """Implement the unwrap state dict helper.

    Args:
        state_dict (Any): State dict value.

    Returns:
        Any: Computed result."""
    for key in ("state", "state_dict", "model_state_dict"):
        if key in state_dict:
            return state_dict[key]
    return state_dict


def _apollo_state_dict_path(model_path):
    """Implement the apollo state dict path helper.

    Args:
        model_path (str | os.PathLike): Model path value.

    Returns:
        Any: Computed result."""
    root, ext = os.path.splitext(model_path)
    candidates = []
    if ext:
        candidates.append(f"{root}.pymss_state_dict.pt")
    candidates.append(f"{model_path}.pymss_state_dict.pt")
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return model_path


def _load_state_dict(model_type, model_path, device):
    """Load state dict.

    Args:
        model_type (Any): Model type value.
        model_path (str | os.PathLike): Model path value.
        device (Any): Device value.

    Returns:
        Any: Computed result."""
    if model_type == "vr":
        return None
    map_location = "cpu"
    if model_type == "htdemucs":
        stubbed_modules = _install_demucs_pickle_stubs()
        try:
            state_dict = torch.load(model_path, map_location=map_location, weights_only=False)
        finally:
            _restore_modules(stubbed_modules)
        return _unwrap_state_dict(state_dict)
    if model_type == "apollo":
        model_path = _apollo_state_dict_path(model_path)
        return _unwrap_state_dict(torch.load(model_path, map_location=map_location, weights_only=False))
    try:
        return _unwrap_state_dict(torch.load(model_path, map_location=map_location, weights_only=True, mmap=True))
    except (TypeError, ValueError, RuntimeError):
        return _unwrap_state_dict(torch.load(model_path, map_location=map_location, weights_only=True))


@contextmanager
def _skip_torch_default_init():
    """Implement the skip torch default init helper.

    Args:
        None: This callable does not accept user-provided arguments.

    Returns:
        None: This callable completes for its side effects."""
    classes = (
        torch.nn.Linear,
        torch.nn.Bilinear,
        torch.nn.Conv1d,
        torch.nn.Conv2d,
        torch.nn.Conv3d,
        torch.nn.ConvTranspose1d,
        torch.nn.ConvTranspose2d,
        torch.nn.ConvTranspose3d,
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.GroupNorm,
        torch.nn.Embedding,
        torch.nn.EmbeddingBag,
        torch.nn.RNN,
        torch.nn.GRU,
        torch.nn.LSTM,
        torch.nn.MultiheadAttention,
    )
    saved = {cls: cls.reset_parameters for cls in classes if hasattr(cls, "reset_parameters")}
    try:
        for cls in saved:
            cls.reset_parameters = lambda self: None
        yield
    finally:
        for cls, reset_parameters in saved.items():
            cls.reset_parameters = reset_parameters


def _install_demucs_pickle_stubs():
    """Implement the install demucs pickle stubs helper.

    Args:
        None: This callable does not accept user-provided arguments.

    Returns:
        Any: Computed result."""
    import sys
    import types

    module_names = ("demucs", "demucs.demucs", "demucs.hdemucs", "demucs.htdemucs")
    previous = {name: sys.modules.get(name) for name in module_names}
    package = sys.modules.setdefault("demucs", types.ModuleType("demucs"))
    package.__path__ = []
    for module_name, class_names in {
        "demucs": ("Demucs",),
        "hdemucs": ("HDemucs", "HTDemucs"),
        "htdemucs": ("HTDemucs",),
    }.items():
        full_name = f"demucs.{module_name}"
        module = sys.modules.setdefault(full_name, types.ModuleType(full_name))
        setattr(package, module_name, module)
        for class_name in class_names:
            if not hasattr(module, class_name):
                setattr(module, class_name, type(class_name, (), {"__module__": full_name}))
    return previous


def _restore_modules(previous):
    """Implement the restore modules helper.

    Args:
        previous (Any): Previous value.

    Returns:
        None: This callable completes for its side effects."""
    import sys

    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _runtime_model_type(model_type, state_dict):
    """Implement the runtime model type helper.

    Args:
        model_type (Any): Model type value.
        state_dict (Any): State dict value.

    Returns:
        Any: Computed result."""
    return "bs_roformer_hyperace" if model_type == "bs_roformer" and any(".segm." in key for key in state_dict) else model_type


def _infer_mel_band_roformer_mlp_hidden_layers(state_dict):
    """Implement the infer mel band roformer mlp hidden layers helper.

    Args:
        state_dict (Any): State dict value.

    Returns:
        Any: Computed result."""
    pattern = re.compile(r"(?:^|\.)mask_estimators\.0\.to_freqs\.0\.0\.(\d+)\.weight$")
    layer_indices = sorted({int(match.group(1)) for key in state_dict for match in [pattern.search(key)] if match})
    if not layer_indices:
        return None
    return len(layer_indices) - 1


def _store_torch_model_on_cpu_for_mlx(config, device):
    """Implement the store torch model on cpu for mlx helper.

    Args:
        config (AttrDict | dict): Loaded pymss configuration.
        device (Any): Device value.

    Returns:
        Any: Computed result."""
    return torch.device(device).type == "mps" and config.inference.get("mps_model_backend", "torch") == "mlx_full"


def _coerce_mps_float64(module):
    """Implement the coerce mps float64 helper.

    Args:
        module (Any): Module value.

    Returns:
        None: This callable completes for its side effects."""
    for child in module.modules():
        for name, param in list(child._parameters.items()):
            if param is not None and param.dtype == torch.float64:
                child._parameters[name] = torch.nn.Parameter(param.detach().float(), requires_grad=param.requires_grad)
        for name, buffer in list(child._buffers.items()):
            if buffer is not None and buffer.dtype == torch.float64:
                child._buffers[name] = buffer.float()


def _coerce_low_precision_to_float32(module):
    """Run CPU and DirectML inference in float32 for low-precision checkpoints."""
    for child in module.modules():
        for name, param in list(child._parameters.items()):
            if param is not None and param.dtype in {torch.float16, torch.bfloat16}:
                child._parameters[name] = torch.nn.Parameter(param.detach().float(), requires_grad=param.requires_grad)
        for name, buffer in list(child._buffers.items()):
            if buffer is not None and buffer.dtype in {torch.float16, torch.bfloat16}:
                child._buffers[name] = buffer.float()


def _normalize_model_dtype(value):
    """Normalize the optional model-parameter precision override."""
    normalized = str(value or "auto").strip().lower().replace("torch.", "")
    aliases = {
        "auto": "auto",
        "fp16": "float16",
        "half": "float16",
        "float16": "float16",
        "fp32": "float32",
        "float": "float32",
        "float32": "float32",
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        raise ValueError("model_dtype must be 'auto', 'float16', or 'float32'") from error


def _model_is_stereo(model_type, config):
    """Implement the model is stereo helper.

    Args:
        model_type (Any): Model type value.
        config (AttrDict | dict): Loaded pymss configuration.

    Returns:
        Any: Computed result."""
    if model_type == "vr":
        return True
    if model_type in ["bs_roformer", "bs_roformer_hyperace", "mel_band_roformer", *LEGACY_DEMUCS_MODEL_TYPES]:
        return config.model.get("stereo", True)
    return True


def _prepare_mix_channels(mix, is_stereo, logger):
    """Implement the prepare mix channels helper.

    Args:
        mix (np.ndarray): Mix value.
        is_stereo (Any): Is stereo value.
        logger (logging.Logger | None): Optional logger for progress messages.

    Returns:
        Any: Computed result."""
    if is_stereo and len(mix.shape) == 1:
        logger.warning("Track is mono, but model is stereo, adding a second channel.")
        return np.stack([mix, mix], axis=0)
    if is_stereo and len(mix.shape) > 2:
        logger.warning("Track has more than 2 channels, taking mean of all channels and adding a second channel.")
        mono = np.mean(mix, axis=0)
        return np.stack([mono, mono], axis=0)
    if not is_stereo and len(mix.shape) != 1:
        logger.warning("Track has more than 1 channels, but model is mono, taking mean of all channels.")
        return np.mean(mix, axis=0)
    return mix


def _standardize_mix(mix, enabled, logger):
    """Implement the standardize mix helper.

    Args:
        mix (np.ndarray): Mix value.
        enabled (Any): Enabled value.
        logger (logging.Logger | None): Optional logger for progress messages.

    Returns:
        Any: Computed result."""
    if not enabled:
        return mix, None

    mono = mix.mean(0)
    mean = mono.mean()
    std = mono.std()
    logger.debug(f"Standardize mix with mean: {mean}, std: {std}")
    return (mix - mean) / std, (mean, std)


def _normalize_outputs(results, enabled, logger, target_peak=OUTPUT_NORMALIZE_PEAK):
    """Normalize outputs.

    Args:
        results (dict): Results value.
        enabled (Any): Enabled value.
        logger (logging.Logger | None): Optional logger for progress messages.
        target_peak (Any, optional): Target peak value. Defaults to OUTPUT_NORMALIZE_PEAK.

    Returns:
        Any: Computed result."""
    if not enabled:
        return results

    peak = max(
        (float(np.max(np.abs(np.asarray(audio)))) for audio in results.values() if np.asarray(audio).size),
        default=0.0,
    )
    if peak <= 0.0 or not np.isfinite(peak):
        logger.debug("Skipping output normalize because peak is zero or not finite.")
        return results

    gain = target_peak / peak
    logger.debug(f"Normalize output stems with peak: {peak}, target_peak: {target_peak}, gain: {gain}")
    return {stem: np.asarray(audio) * gain for stem, audio in results.items()}


def _destandardize(estimates, stats):
    """Implement the destandardize helper.

    Args:
        estimates (Any): Estimates value.
        stats (Any): Stats value.

    Returns:
        Any: Computed result."""
    return estimates if stats is None else estimates * stats[1] + stats[0]


def _tta_variants(mix, use_tta, logger):
    """Implement the tta variants helper.

    Args:
        mix (np.ndarray): Mix value.
        use_tta (Any): Use tta value.
        logger (logging.Logger | None): Optional logger for progress messages.

    Returns:
        Any: Computed result."""
    if not use_tta:
        return [mix.copy()]
    variants = [mix.copy(), mix[::-1].copy(), -1.0 * mix.copy()]
    logger.debug(f"User needs to apply TTA, total tracks: {len(variants)}")
    return variants


def _merge_tta_results(results):
    """Implement the merge tta results helper.

    Args:
        results (dict): Results value.

    Returns:
        Any: Computed result."""
    waveforms = results[0]
    for index, result in enumerate(results[1:], start=1):
        for stem, audio in result.items():
            waveforms[stem] += audio[::-1].copy() if index == 1 else -1.0 * audio

    for stem in waveforms:
        waveforms[stem] /= len(results)
    return waveforms


def _build_results(waveforms, instruments, mix_orig, config, standardize_stats, logger):
    """Build results.

    Args:
        waveforms (Any): Waveforms value.
        instruments (Sequence[str] | None): Instruments value.
        mix_orig (Any): Mix orig value.
        config (AttrDict | dict): Loaded pymss configuration.
        standardize_stats (Any): Standardize stats value.
        logger (logging.Logger | None): Optional logger for progress messages.

    Returns:
        Any: Built value."""
    target_instrument = config.training.target_instrument
    if target_instrument is None:
        return {instr: _destandardize(waveforms[instr].T, standardize_stats) for instr in instruments}

    results = {}
    target_audio = _destandardize(waveforms[target_instrument].T, standardize_stats)
    if target_instrument in instruments:
        results[target_instrument] = target_audio
    other_instruments = [instr for instr in config.training.instruments if instr != target_instrument]
    logger.debug(
        f"target_instrument is not null, extracting instrumental from {target_instrument}, other_instruments: {other_instruments}"
    )
    if other_instruments:
        secondary = other_instruments[0]
        if secondary in instruments:
            results[secondary] = mix_orig.T - target_audio
    return results


def _resolve_instruments(config, stems=None):
    """Resolve instruments.

    Args:
        config (AttrDict | dict): Loaded pymss configuration.
        stems (Sequence[str] | None, optional): Requested output stem names. Defaults to None.

    Returns:
        Any: Resolved value."""
    instruments = config.training.instruments.copy()
    if stems is None:
        source_indices = None if config.training.target_instrument is None else (0,)
        return instruments, source_indices

    stem_list = [stems] if isinstance(stems, str) else list(stems)
    lower_to_index = {instr.lower(): index for index, instr in enumerate(instruments)}
    selected, indices = [], []
    for stem in stem_list:
        key = stem.lower()
        if key not in lower_to_index:
            raise ValueError(f"Invalid instrument key: {stem}. Valid instrument keys: {instruments}")
        index = lower_to_index[key]
        if index in indices:
            continue
        selected.append(instruments[index])
        indices.append(index)
    if not selected:
        raise ValueError("stems must not be empty")
    source_indices = tuple(indices) if config.training.target_instrument is None else (0,)
    return selected, source_indices


def _get_store_dir(store_dirs, instr):
    """Return store dir.

    Args:
        store_dirs (Any): Store dirs value.
        instr (Any): Instr value.

    Returns:
        Any: Computed result."""
    if instr in store_dirs:
        return store_dirs[instr]
    instr_lower = instr.lower()
    for key, value in store_dirs.items():
        if key.lower() == instr_lower:
            return value
    return ""


def _as_store_path(value):
    """Return a filesystem path value as a string, or None for unsupported values."""
    if isinstance(value, str):
        return value
    return None


def _iter_store_paths(value):
    """Yield filesystem paths from one ``store_dirs`` value."""
    if isinstance(value, list):
        for item in value:
            path = _as_store_path(item)
            if path:
                yield path
        return

    path = _as_store_path(value)
    if path:
        yield path


def _has_single_store_folder(store_dirs):
    """Return whether ``store_dirs`` routes saved stems to one folder."""
    if _as_store_path(store_dirs):
        return True
    if not isinstance(store_dirs, dict):
        return False

    folders = set()
    for value in store_dirs.values():
        paths = list(_iter_store_paths(value))
        if not paths:
            return False
        folders.update(os.path.normcase(os.path.abspath(path)) for path in paths)
    return len(folders) == 1


class MSSeparator:
    """Load a music source separation model and run inference.

    ``MSSeparator`` is the main Python API for pymss. Prefer
    ``MSSeparator.from_model_name(...)`` for catalog models; use the
    constructor directly when you have custom weights, a custom YAML config,
    or need full control over runtime parameters.

    Args:
        model_type (str): Model architecture/runtime type. Common values
            include ``bs_roformer``, ``mel_band_roformer``, ``htdemucs``,
            ``mdx23c``, ``bandit``, ``bandit_v2``, ``scnet``, ``apollo``,
            ``vr``, ``legacy_demucs``, and ``legacy_tasnet``.
        model_path (str | os.PathLike): Path to the model weights file, such
            as a ``.ckpt``, ``.th``, ``.pth``, or VR model file.
        config_path (str | os.PathLike | None, optional): YAML config path for
            MSS-style models. If omitted, pymss tries ``model_path + ".yaml"``.
            VR models use built-in metadata instead of an MSS YAML config.
            Defaults to None.
        device (str, optional): Runtime device. Valid values are ``auto``,
            ``cpu``, ``cuda``, ``mps``, and ``mlx``. ``auto`` chooses CUDA
            first, then Apple MPS, then CPU. ``mlx`` is a public shortcut for
            Apple Silicon MLX execution through the MPS device path. Defaults
            to ``"auto"``.
        device_ids (list[int], optional): CUDA device IDs. Multiple IDs can
            enable ``torch.nn.DataParallel`` for supported Torch models. This
            does not select multiple MPS or MLX devices. Defaults to ``[0]``.
        output_format (str, optional): Format used by ``process_folder()`` and
            ``save_audio()``. Supported values are ``wav``, ``flac``, ``mp3``,
            and ``m4a``. Defaults to ``"wav"``.
        use_tta (bool, optional): Enables test-time augmentation. It can
            improve quality for some MSS models but increases inference time.
            Defaults to False.
        store_dirs (str | dict, optional): Output routing for saved stems. A
            string writes every saved stem to the same folder. A dict maps stem
            names to a folder, a list of folders, ``None``, or an empty value.
            Missing/empty values skip that stem. Defaults to ``"results"``.
        save_as_folder (bool, optional): When True and ``store_dirs`` resolves
            to one output folder, each input audio file is saved into its own
            subfolder named after the input audio basename. Defaults to False.
        audio_params (dict, optional): Encoding options used only when writing
            files, for example ``wav_bit_depth``, ``flac_bit_depth``,
            ``mp3_bit_rate``, ``m4a_bit_rate``, ``m4a_codec``, and
            ``m4a_aac_at_quality``.
        logger (logging.Logger | None, optional): Logger instance. If omitted,
            pymss uses ``pymss.get_separation_logger()``. Defaults to None.
        debug (bool, optional): Enables debug logging and disables some normal
            progress-bar behavior. Defaults to False.
        progress_callback (callable | None, optional): Optional callback used
            by lower-level demixing code. It receives progress information
            during long-running inference. Demix progress is reported as
            ``callback(done_seconds, total_seconds, message)``. Defaults to
            None.
        inference_params (dict, optional): Runtime inference overrides. Common
            keys include ``batch_size``, ``overlap_size``, ``chunk_size``,
            ``stem_batch_size``, ``standardize``, ``normalize``, ``mask_mode``,
            attention backend options, and VR-specific options such as
            ``aggression`` and ``window_size``.

    Example:
        >>> separator = MSSeparator.from_model_name(
        ...     "bs_roformer_voc_hyperacev2",
        ...     download=True,
        ...     model_dir="models",
        ...     output_format="wav",
        ...     inference_params={"standardize": None, "normalize": False},
        ... )
        >>> separator.process_folder("song.wav")

    Example:
        >>> separator = MSSeparator(
        ...     model_type="mel_band_roformer",
        ...     model_path="models/custom.ckpt",
        ...     config_path="models/custom.yaml",
        ...     device="cuda",
        ...     store_dirs={
        ...         "vocals": "out/vocals",
        ...         "instrumental": ["out/instrumental", "backup/instrumental"],
        ...         "drums": None,
        ...     },
        ...     inference_params={"standardize": True, "normalize": True},
        ... )"""

    def __init__(
        self,
        model_type,
        model_path,
        config_path=None,
        device="auto",
        device_ids=[0],
        output_format="wav",
        use_tta=False,
        store_dirs="results",  # str for single folder, dict with instrument keys for multiple folders
        save_as_folder=False,
        audio_params={
            "wav_bit_depth": "FLOAT",
            "flac_bit_depth": "PCM_24",
            "mp3_bit_rate": "320k",
            "m4a_bit_rate": "512k",
            "m4a_codec": "aac",
            "m4a_aac_at_quality": 2,
        },
        logger=None,
        debug=False,
        progress_callback=None,
        inference_params={
            "batch_size": None,
            "overlap_size": None,
            "chunk_size": None,
            "standardize": None,
            "normalize": False,
            "mask_mode": None,
        },
    ):
        """Initialize and load a separator from explicit model files.

        Args:
            model_type (str): Runtime model family. Catalog users usually get
                this value from ``MSSeparator.from_model_name()`` instead of
                setting it manually.
            model_path (str | os.PathLike): Model weights path.
            config_path (str | os.PathLike | None, optional): YAML config path.
                If omitted, pymss tries ``model_path + ".yaml"``. Defaults to
                None.
            device (str, optional): ``auto``, ``cpu``, ``cuda``, ``mps``, or
                ``mlx``. Defaults to ``"auto"``.
            device_ids (list[int], optional): CUDA device IDs used when CUDA
                and DataParallel are available. Defaults to ``[0]``.
            output_format (str, optional): Saved audio format: ``wav``,
                ``flac``, ``mp3``, or ``m4a``. Defaults to ``"wav"``.
            use_tta (bool, optional): Enables test-time augmentation. Defaults
                to False.
            store_dirs (str | dict, optional): Folder routing for saved stems.
                For example ``"results"`` saves all stems to one folder, while
                ``{"vocals": "out/vocals", "drums": None}`` saves only
                vocals and skips drums. Defaults to ``"results"``.
            save_as_folder (bool, optional): If True and ``store_dirs`` is a
                single folder path, or every saved dict destination resolves to
                the same folder, ``process_folder()`` writes each input file's
                stems under ``<output>/<audio_name>/``. Defaults to False.
            audio_params (dict, optional): Encoder settings. Examples:
                ``{"wav_bit_depth": "FLOAT"}``,
                ``{"flac_bit_depth": "PCM_24"}``,
                ``{"mp3_bit_rate": "320k"}``, or
                ``{"m4a_codec": "aac", "m4a_bit_rate": "512k"}``.
            logger (logging.Logger | None, optional): Logger to use. Defaults
                to None.
            debug (bool, optional): Enables verbose debug logging. Defaults to
                False.
            progress_callback (callable | None, optional): Progress callback
                passed into demixing helpers. Demix progress is reported as
                ``callback(done_seconds, total_seconds, message)``. Defaults
                to None.
            inference_params (dict, optional): Inference overrides. ``None``
                values keep model config defaults. ``standardize`` controls
                legacy input standardization, and ``normalize`` controls linked
                output peak normalization.

        Returns:
            None: The separator is loaded and ready for inference.

        Example:
            >>> separator = MSSeparator(
            ...     model_type="htdemucs",
            ...     model_path="models/htdemucs.th",
            ...     config_path="models/htdemucs.yaml",
            ...     inference_params={"chunk_size": 485100, "normalize": True},
            ... )"""
        if not model_type:
            raise ValueError("model_type is required")
        if not model_path:
            raise ValueError("model_path is required")

        logger = logger if logger is not None else get_separation_logger()
        device, inference_params = _resolve_public_device(device, inference_params, logger)

        self.model_type = model_type

        self.model_path = model_path
        self.config_path_given = config_path is not None
        self.config_path = config_path if config_path else (model_path + ".yaml")
        self.output_format = output_format
        self.use_tta = use_tta
        self.store_dirs = store_dirs
        self.save_as_folder = save_as_folder
        self.audio_params = audio_params
        self.logger = logger
        self.debug = debug
        self.progress_callback = progress_callback
        self.inference_params = inference_params
        self.output_normalize = self.inference_params.get("normalize", False)

        if self.debug:
            set_log_level(self.logger, logging.DEBUG)
        else:
            set_log_level(self.logger, logging.INFO)

        self.log_system_info()
        self.check_ffmpeg_installed()

        self.device_ids = device_ids
        self.device = _select_device(device, self.device_ids, self.logger)
        self.inference_params = _prefer_mlx_for_auto(device, self.device, self.inference_params, self.logger)

        self._cudnn_benchmark_initial = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = True
        self.logger.info(f"Using device: {self.device}, device_ids: {self.device_ids}")

        self.model, self.config = self.load_model()

        if isinstance(self.store_dirs, str):
            self.store_dirs = {k: self.store_dirs for k in self.config.training.instruments}

        valid_instruments = {instr.lower() for instr in self.config.training.instruments}
        for key in list(self.store_dirs.keys()):
            if key not in self.config.training.instruments and key.lower() not in valid_instruments:
                self.store_dirs.pop(key)
                self.logger.warning(f"Invalid instrument key: {key}, removing from store_dirs")
                self.logger.warning(f"Valid instrument keys: {self.config.training.instruments}")

        self.save_as_folder = bool(self.save_as_folder and _has_single_store_folder(self.store_dirs))

    def __enter__(self):
        """Return the loaded separator when entering a ``with`` block.

        Args:
            None: This callable does not accept user-provided arguments.

        Returns:
            MSSeparator: This separator instance.

        Example:
            >>> with MSSeparator.from_model_name("bs_roformer_voc_hyperacev2") as separator:
            ...     separator.process_folder("song.wav")"""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the separator when leaving a ``with`` block.

        Args:
            exc_type (type[BaseException] | None): Exception type raised inside
                the ``with`` block, or None when the block exits normally.
            exc_value (BaseException | None): Exception instance raised inside
                the ``with`` block, or None when the block exits normally.
            traceback (types.TracebackType | None): Traceback for the exception,
                or None when the block exits normally.

        Returns:
            bool: False, so exceptions raised inside the ``with`` block are not
            suppressed.

        Example:
            >>> with MSSeparator.from_model_name("bs_roformer_voc_hyperacev2") as separator:
            ...     results = separator.separate(audio)"""
        self.close()
        return False

    @classmethod
    def from_model_name(cls, model_name, model_dir=None, download=False, source="modelscope", endpoint=None, **kwargs):
        """Create a separator from a model catalog name or alias.

        This resolves the model type, weights path, config path, and auxiliary
        files from the pymss model catalog, then forwards remaining keyword
        arguments to ``MSSeparator(...)``.

        Args:
            model_name (str): Catalog model name or alias, for example
                ``"bs_roformer_voc_hyperacev2"``.
            model_dir (str | os.PathLike | None, optional): Directory used to
                find or download model files. Uses the default pymss cache when
                omitted. Defaults to None.
            download (bool, optional): If True, missing model files are
                downloaded before loading. If False, missing files raise
                ``FileNotFoundError``. Defaults to False.
            source (str, optional): Download source passed to the downloader:
                ``modelscope``, ``huggingface``, or ``hf-mirror``. Defaults to
                ``"modelscope"``.
            endpoint (str | None, optional): Optional custom file-serving
                endpoint. Defaults to None.
            **kwargs: Extra arguments forwarded to ``MSSeparator(...)``, such
                as ``device``, ``output_format``, ``store_dirs``,
                ``save_as_folder``, ``audio_params``, ``debug``, and
                ``inference_params``.

        Returns:
            MSSeparator: Loaded separator instance.

        Example:
            >>> separator = MSSeparator.from_model_name(
            ...     "bs_roformer_voc_hyperacev2",
            ...     download=True,
            ...     model_dir="models",
            ...     device="auto",
            ...     output_format="flac",
            ...     inference_params={"normalize": True},
            ... )"""
        if download:
            from .model_download import download_model

            download_model(model_name, model_dir=model_dir, source=source, endpoint=endpoint)

        from .model_registry import resolve_model

        resolved = resolve_model(model_name, model_dir=model_dir, require_supported=True, require_exists=True)
        return cls(
            model_type=resolved["model_type"],
            model_path=resolved["model_path"],
            config_path=resolved["config_path"],
            **kwargs,
        )

    def log_system_info(self):
        """Log runtime system information at debug level.

        Args:
            None: This callable does not accept user-provided arguments.

        Returns:
            None: Operating system, Python, and PyTorch versions are logged."""
        os_name = platform.system()
        os_version = platform.version()
        self.logger.debug(f"Operating System: {os_name} {os_version}")

        python_version = platform.python_version()
        self.logger.debug(f"Python Version: {python_version}")

        pytorch_version = torch.__version__
        self.logger.debug(f"PyTorch Version: {pytorch_version}")

    def check_ffmpeg_installed(self):
        """Check whether the ``ffmpeg`` executable is available.

        Args:
            None: This callable does not accept user-provided arguments.

        Returns:
            None: A warning is logged when ffmpeg cannot be found."""
        try:
            ffmpeg_version_output = subprocess.check_output(["ffmpeg", "-version"], text=True)
            first_line = ffmpeg_version_output.splitlines()[0]
            self.logger.debug(f"FFmpeg installed: {first_line}")
        except FileNotFoundError:
            self.logger.warning("FFmpeg is not installed. Please install FFmpeg to use this package.")

    def load_model(self):
        """Load model weights and build the runtime config.

        Args:
            None: This callable does not accept user-provided arguments.

        Returns:
            tuple[torch.nn.Module | object, AttrDict]: Loaded model runtime and
            resolved configuration. VR models return a VR runtime object;
            MSS-style models return a Torch module.

        Notes:
            VR models are initialized from built-in metadata. MSS-style models
            load the YAML config, apply ``inference_params``, configure optional
            attention/model backends, then load the state dict."""
        start_time = time()
        if self.model_type == "vr":
            from .modules.vocal_remover.vr_models import get_vr_model_metadata
            from .modules.vocal_remover import VRSeparator

            model_data = get_vr_model_metadata(self.model_path)
            instruments = [model_data["primary_stem"], model_data["secondary_stem"]]
            config = AttrDict(
                {
                    "training": {
                        "instruments": instruments,
                        "target_instrument": None,
                        "use_amp": True,
                    },
                    "audio": {
                        "sample_rate": 44100,
                    },
                    "inference": {
                        "batch_size": 2,
                        "window_size": 512,
                        "aggression": 5,
                        "enable_tta": self.use_tta,
                        "enable_post_process": False,
                        "post_process_threshold": 0.2,
                        "high_end_process": False,
                        "use_amp": True,
                        "fuse_conv_bn": False,
                        "use_channels_last": False,
                        "standardize": False,
                        "normalize": False,
                    },
                }
            )
            self.update_inference_params(config, self.inference_params)
            common_config = {
                "logger": self.logger,
                "debug": self.debug,
                "torch_device": self.device,
                "torch_device_cpu": torch.device("cpu"),
                "torch_device_mps": torch.device("mps") if torch.device(self.device).type == "mps" else None,
                "model_name": os.path.basename(self.model_path),
                "model_path": self.model_path,
                "model_data": model_data,
                "sample_rate": 44100,
                "progress_callback": self.progress_callback,
            }
            model = VRSeparator(common_config, config.inference)
            model.load_model()
            self._log_model_config("vr", config, include_config_path=False)
            self.logger.debug(f"Loading VR model completed, duration: {time() - start_time:.2f} seconds")
            return model, config

        if self.model_type in LEGACY_DEMUCS_MODEL_TYPES:
            from .modules.legacy_demucs import load_legacy_demucs_model

            config_path = self.config_path if self.config_path_given else None
            model, config = load_legacy_demucs_model(self.model_path, config_path)
            config = AttrDict(config)
            self.update_inference_params(config, self.inference_params)
            model = model.to(self.device)
            model.eval()

            self._log_model_config(self.model_type, config, config_path=config_path)
            self.logger.debug(f"Loading legacy Demucs/TasNet model completed, duration: {time() - start_time:.2f} seconds")
            return model, config

        state_dict = _load_state_dict(self.model_type, self.model_path, self.device)
        model_type = _runtime_model_type(self.model_type, state_dict)
        model_kwargs_override = None
        if model_type == "mel_band_roformer":
            model_kwargs_override = {
                "mlp_hidden_layers": _infer_mel_band_roformer_mlp_hidden_layers(state_dict),
            }

        init_context = _skip_torch_default_init() if model_type in FAST_INIT_MODEL_TYPES else nullcontext()
        with init_context:
            model, config = get_model_from_config(model_type, self.config_path, model_kwargs_override=model_kwargs_override)

        self.update_inference_params(config, self.inference_params)
        self.apply_model_inference_config(model, config)

        self._log_model_config(model_type, config, config_path=self.config_path)

        try:
            model.load_state_dict(state_dict, assign=True)
        except TypeError:
            model.load_state_dict(state_dict)
        device_type = torch.device(self.device).type
        model_dtype = _normalize_model_dtype(config.inference.get("model_dtype", "auto"))
        if device_type == "mps":
            _coerce_mps_float64(model)
        if device_type == "cpu":
            _coerce_low_precision_to_float32(model)
        elif device_type == "privateuseone":
            if model_dtype == "float16":
                if model_type not in DML_FP16_MODEL_TYPES:
                    raise ValueError(
                        "DirectML float16 is supported only for BS-Roformer and Mel-Band-Roformer models"
                    )
                model.half()
            else:
                # Generic API calls keep the historical FP32 behavior for
                # model_dtype=auto. The RVC worker resolves auto explicitly and
                # can retry a failed FP16 attempt in a fresh FP32 process.
                _coerce_low_precision_to_float32(model)

        keep_torch_model_cpu = _store_torch_model_on_cpu_for_mlx(config, self.device)
        if torch.device(self.device).type == "cuda" and len(self.device_ids) > 1 and not keep_torch_model_cpu:
            model = torch.nn.DataParallel(model, device_ids=self.device_ids)
        model = model.to("cpu" if keep_torch_model_cpu else self.device)
        model.eval()

        self.logger.debug(f"Loading model completed, duration: {time() - start_time:.2f} seconds")
        return model, config

    def _log_model_config(self, model_type, config, config_path=None, include_config_path=True):
        """Log resolved separator, audio, and model inference settings.

        Args:
            model_type (str): Runtime model type being loaded.
            config (AttrDict | dict): Loaded model configuration.
            config_path (str | os.PathLike | None, optional): Config path to
                include in logs. Defaults to None.
            include_config_path (bool, optional): Whether to include
                ``config_path`` in the separator log line. Defaults to True.

        Returns:
            None: Model settings are written to the logger."""
        config_path_part = f", config_path: {config_path}" if include_config_path else ""
        self.logger.info(
            f"Separator params: model_type: {model_type}, model_path: {self.model_path}{config_path_part}, output_folder: {self.store_dirs}, save_as_folder: {self.save_as_folder}"
        )
        self.logger.info(f"Audio params: output_format: {self.output_format}, audio_params: {self.audio_params}")
        self.logger.info(
            f"Model params: instruments: {config.training.get('instruments', None)}, target_instrument: {config.training.get('target_instrument', None)}"
        )
        self.logger.info(
            f"Model params: batch_size: {config.inference.get('batch_size', None)}, standardize: {config.inference.get('normalize', None)}, normalize: {self.output_normalize}, use_tta: {self.use_tta}"
        )
        if model_type == "vr":
            self.logger.info(
                f"VR model params: window_size: {config.inference.get('window_size', None)}, aggression: {config.inference.get('aggression', None)}, enable_tta: {config.inference.get('enable_tta', None)}, enable_post_process: {config.inference.get('enable_post_process', None)}, post_process_threshold: {config.inference.get('post_process_threshold', None)}, high_end_process: {config.inference.get('high_end_process', None)}"
            )
            self.logger.debug(
                f"VR model params: use_amp: {config.inference.get('use_amp', None)}, fuse_conv_bn: {config.inference.get('fuse_conv_bn', None)}, use_channels_last: {config.inference.get('use_channels_last', None)}"
            )
        else:
            self.logger.info(
                f"MSS model params: chunk_size: {config.audio.get('chunk_size', None)}, overlap_size: {config.inference.get('overlap_size', None)}, stem_batch_size: {config.inference.get('stem_batch_size', None)}"
            )
            self.logger.debug(
                f"MSS model params: mask_mode: {config.inference.get('mask_mode', None)}, model_dtype: {config.inference.get('model_dtype', 'auto')}, cuda_attention_backend: {config.inference.get('cuda_attention_backend', None)}, mps_attention_backend: {config.inference.get('mps_attention_backend', None)}, mps_mlx_min_tokens: {config.inference.get('mps_mlx_min_tokens', None)}, mps_model_backend: {config.inference.get('mps_model_backend', None)}, mps_model_compute_dtype: {config.inference.get('mps_model_compute_dtype', None)}"
            )

    def apply_model_inference_config(self, model, config):
        """Apply config-driven runtime options to a loaded model.

        Args:
            model (torch.nn.Module): Loaded model instance.
            config (AttrDict | dict): Loaded pymss configuration containing
                inference options such as ``mask_mode``,
                ``cuda_attention_backend``, ``mps_attention_backend``,
                ``mps_mlx_min_tokens``, ``mps_model_backend``, and
                ``mps_model_compute_dtype``.

        Returns:
            None: Supported options are applied directly to model modules."""
        if hasattr(model, "set_mask_mode"):
            model.set_mask_mode(config.inference.get("mask_mode", "no_segm"))
        cuda_attention_backend = config.inference.get("cuda_attention_backend", None)
        if cuda_attention_backend is not None:
            for module in model.modules():
                if hasattr(module, "set_cuda_attention_backend"):
                    module.set_cuda_attention_backend(cuda_attention_backend)
        model_backend = config.inference.get("mps_model_backend", None)
        if model_backend is not None:
            compute_dtype = config.inference.get("mps_model_compute_dtype", None)
            for module in model.modules():
                if hasattr(module, "set_mps_model_backend"):
                    module.set_mps_model_backend(model_backend, compute_dtype)
        backend = config.inference.get("mps_attention_backend", None)
        min_tokens = config.inference.get("mps_mlx_min_tokens", 128)
        if backend is not None:
            for module in model.modules():
                if hasattr(module, "set_mps_attention_backend"):
                    module.set_mps_attention_backend(backend, min_tokens)

    def update_inference_params(self, config, params):
        # Keep this mapping explicit:
        # public/API/CLI "standardize" controls legacy input standardization performed by _standardize_mix().
        # existing MSS YAML files still store that switch as inference.normalize, and those YAML files cannot be renamed in place without breaking compatibility.
        # public/API/CLI "normalize" is a separate output peak normalization option stored in self.output_normalize, so it must not overwrite config.inference["normalize"] here.
        """Apply user inference overrides to a loaded config.

        Args:
            config (AttrDict | dict): Loaded model configuration. For MSS YAML
                models, ``config.inference.normalize`` is the legacy input
                standardization key.
            params (dict | None): Runtime overrides. Keys with ``None`` keep
                the model config value. ``standardize`` maps to the legacy YAML
                ``inference.normalize`` key, while ``normalize`` is stored on
                ``self.output_normalize`` and is not written into the YAML
                compatibility key.

        Returns:
            AttrDict | dict: The same config object after applying overrides.

        Example:
            >>> separator.update_inference_params(
            ...     config,
            ...     {"batch_size": 2, "standardize": True, "normalize": True},
            ... )

        Notes:
            ``standardize=True`` standardizes the input mix before inference and
            restores scale afterward. ``normalize=True`` peak-normalizes the
            selected output stems together after separation."""
        if "normalize" not in config.inference:
            config.inference["normalize"] = False
        standardize = params.get("standardize")
        if standardize is not None:
            config.inference["normalize"] = standardize

        for key, section in INFERENCE_PARAM_TARGETS.items():
            if key in {"standardize", "normalize"}:
                continue
            value = params.get(key)
            if value is None:
                continue
            if key not in PASSTHROUGH_INFERENCE_PARAMS:
                value = float(value) if key in {"post_process_threshold", "overlap"} else int(value)
            config[section][key] = value
        return config

    def _save_output(self, instr, audio, sr, file_name, save_dir):
        """Save one separated stem to one output directory.

        Args:
            instr (str): Stem/instrument name appended to the output filename.
            audio (np.ndarray): Stem audio samples.
            sr (int): Sample rate.
            file_name (str): Base input filename without extension.
            save_dir (str): Destination directory. When ``save_as_folder`` is
                active, this is already the per-input audio subfolder.

        Returns:
            None: The stem is written to disk."""
        output_format = self.output_format.lower()
        os.makedirs(save_dir, exist_ok=True)
        self.save_audio(audio, sr, f"{file_name}_{instr}", save_dir)
        self.logger.debug(f"Saved {instr} for {file_name}_{instr}.{output_format} in {save_dir}")

    def _resolve_output_dir(self, save_dir, file_name):
        """Return the final folder used for one input audio file.

        Args:
            save_dir (str): Configured output directory.
            file_name (str): Base input filename without extension.

        Returns:
            str: Configured directory, or a per-audio subfolder when
            ``save_as_folder`` is active."""
        return os.path.join(save_dir, file_name) if self.save_as_folder else save_dir

    def _wait_save_futures(self, path, futures):
        """Wait for asynchronous save jobs and report failures.

        Args:
            path (str | os.PathLike): Input track path used in warning logs.
            futures (Iterable[concurrent.futures.Future]): Save jobs returned
                by the save thread pool.

        Returns:
            bool: True when every save job completed successfully."""
        save_ok = True
        for future in futures:
            try:
                future.result()
            except Exception as e:
                save_ok = False
                self.logger.warning(f"Cannot save track: {path}, error: {str(e)}")
        return save_ok

    @staticmethod
    def _submit_load(load_executor, paths, index, sample_rate):
        """Submit the next audio load job to the load executor.

        Args:
            load_executor (ThreadPoolExecutor): Executor used for audio reads.
            paths (Sequence[str | os.PathLike]): Input audio paths.
            index (int): Index of the path to submit.
            sample_rate (int): Target sample rate in Hz.

        Returns:
            concurrent.futures.Future | None: Future for the submitted load, or
            None when ``index`` is outside ``paths``."""
        return None if index >= len(paths) else load_executor.submit(load_audio, paths[index], sr=sample_rate, mono=False)

    def _submit_save_outputs(self, save_executor, results, sr, file_name):
        """Submit save jobs for all returned stems.

        Args:
            save_executor (ThreadPoolExecutor): Executor used for file writes.
            results (dict[str, np.ndarray]): Mapping of stem name to audio.
            sr (int): Sample rate.
            file_name (str): Base input filename without extension. Also used
                as the per-input output folder name when ``save_as_folder`` is
                active.

        Returns:
            list[concurrent.futures.Future]: Save job futures."""
        return [
            save_executor.submit(self._save_output, instr, audio, sr, file_name, output_dir)
            for instr, audio in results.items()
            for save_dir in [_get_store_dir(self.store_dirs, instr)]
            if save_dir
            for configured_output_dir in (save_dir if isinstance(save_dir, list) else [save_dir])
            for output_dir in [self._resolve_output_dir(configured_output_dir, file_name)]
        ]

    def _stems_to_save(self):
        """Return stems that should be saved according to ``store_dirs``.

        Args:
            None: This callable does not accept user-provided arguments.

        Returns:
            list[str] | None: Stem names to request from separation. ``None``
            means all stems should be separated."""
        stems = [instr for instr in self.config.training.instruments if _get_store_dir(self.store_dirs, instr)]
        return stems or None

    def _stem_batches_to_save(self):
        """Return stem groups used by ``process_folder()``.

        Args:
            None: This callable does not accept user-provided arguments.

        Returns:
            list[list[str] | None]: Stem groups. ``stem_batch_size`` can split
            stems into smaller inference groups to reduce memory use. When
            output ``normalize=True``, all saved stems are kept in one group so
            they share the same normalization gain."""
        stems = self._stems_to_save()
        if stems is None:
            return [None]
        if self.output_normalize:
            return [stems]  # we need to normalize across all stems together
        batch_size = int(self.config.inference.get("stem_batch_size", 0))
        if batch_size <= 0 or len(stems) <= batch_size:
            return [stems]
        return [stems[index : index + batch_size] for index in range(0, len(stems), batch_size)]

    def _drain_save_queue(self, pending_saves, success_files, progress, max_pending_saves=0, record_success=True):
        """Drain completed save batches until the queue is small enough.

        Args:
            pending_saves (collections.deque): Queue of ``(path, futures)``
                save batches.
            success_files (list[str]): Processed filenames to update.
            progress (tqdm | None): Optional progress bar.
            max_pending_saves (int, optional): Stop draining once this many
                batches remain. Defaults to 0.
            record_success (bool, optional): Whether to append successfully
                saved filenames to ``success_files``. Defaults to True.

        Returns:
            bool: True when drained save batches completed successfully."""
        ok = True
        while len(pending_saves) > max_pending_saves:
            saved_path, saved_futures = pending_saves.popleft()
            saved_ok = self._wait_save_futures(saved_path, saved_futures)
            ok = saved_ok and ok
            if saved_ok and record_success:
                success_files.append(os.path.basename(saved_path))
                if progress is not None:
                    progress.update(1)
        return ok

    def _wait_pending_saves(self, pending_saves):
        """Wait for every queued save batch to finish.

        Args:
            pending_saves (collections.deque): Queue of ``(path, futures)``
                save batches.

        Returns:
            bool: True when every queued batch saved successfully."""
        ok = True
        while pending_saves:
            saved_path, saved_futures = pending_saves.popleft()
            ok = self._wait_save_futures(saved_path, saved_futures) and ok
        return ok

    def process_folder(self, input_folder):
        """Separate one audio file or every direct file in a folder.

        Args:
            input_folder (str | os.PathLike): Input audio file path or input
                folder. Folder processing considers only direct child files and
                does not recursively walk subfolders.

        Returns:
            list[str]: Basenames of input files that were successfully
            separated and saved.

        Example:
            >>> success_files = separator.process_folder("songs")

        Example:
            >>> separator = MSSeparator.from_model_name(
            ...     "some_six_stem_model",
            ...     store_dirs={"vocals": "out/vocals", "drums": "out/drums"},
            ...     inference_params={"normalize": True},
            ... )
            >>> separator.process_folder("input.wav")

        Notes:
            ``store_dirs`` controls which stems are saved. If only ``vocals``
            and ``drums`` are routed, only those stems are requested and saved.
            When ``save_as_folder=True`` and all saved stems share one output
            folder, stems for ``song.wav`` are written under ``<output>/song/``.
            With output ``normalize=True``, those selected stems share one peak
            normalization gain."""
        if os.path.isfile(input_folder):
            all_mixtures_path = [input_folder]
            input_label = "Input_file"
        elif os.path.isdir(input_folder):
            all_mixtures_path = [os.path.join(input_folder, f) for f in os.listdir(input_folder)]
            input_label = "Input_folder"
        else:
            raise ValueError(f"Input path '{input_folder}' does not exist.")

        if not all_mixtures_path:
            return []

        sample_rate = 44100
        if "sample_rate" in self.config.audio:
            sample_rate = self.config.audio["sample_rate"]
        self.logger.info(
            f"{input_label}: {input_folder}, Total files found: {len(all_mixtures_path)}, Use sample rate: {sample_rate}"
        )

        success_files, pending_saves = [], deque()

        progress = tqdm(all_mixtures_path, desc="Total progress") if not self.debug else None
        try:
            with (
                ThreadPoolExecutor(max_workers=1, thread_name_prefix="pymss-load") as load_executor,
                ThreadPoolExecutor(max_workers=2, thread_name_prefix="pymss-save") as save_executor,
            ):
                load_future = self._submit_load(load_executor, all_mixtures_path, 0, sample_rate)

                for index, path in enumerate(all_mixtures_path):
                    mix = None
                    if progress is not None:
                        progress.set_postfix({"track": os.path.basename(path)})

                    try:
                        mix, sr = load_future.result()
                    except Exception as e:
                        self.logger.warning(f"Cannot process track: {path}, error: {str(e)}")
                        load_future = self._submit_load(load_executor, all_mixtures_path, index + 1, sample_rate)
                        continue

                    load_future = self._submit_load(load_executor, all_mixtures_path, index + 1, sample_rate)

                    self.logger.debug(f"Starting separation process for audio_file: {path}")
                    try:
                        file_name, _ = os.path.splitext(os.path.basename(path))
                        track_saves = deque()
                        save_ok = True
                        for stems in self._stem_batches_to_save():
                            results = self.separate(mix, pbar=False, stems=stems)
                            track_saves.append((path, self._submit_save_outputs(save_executor, results, sr, file_name)))
                            save_ok = (
                                self._drain_save_queue(track_saves, success_files, None, 1, record_success=False) and save_ok
                            )
                            del results
                        save_ok = self._wait_pending_saves(track_saves) and save_ok
                    except Exception as e:
                        self.logger.warning(f"Cannot separate track: {path}, error: {str(e)}")
                        if mix is not None:
                            del mix
                        continue

                    self.logger.debug(f"Separation audio_file: {path} completed. Starting to save results.")
                    if save_ok:
                        success_files.append(os.path.basename(path))
                        if progress is not None:
                            progress.update(1)

                    if mix is not None:
                        del mix

                self._drain_save_queue(pending_saves, success_files, progress)
        finally:
            if progress is not None:
                progress.close()
        return success_files

    def separate(self, mix, pbar=True, stems=None):
        """Run separation on an already loaded audio array.

        Args:
            mix (np.ndarray): Input waveform. Mono and stereo arrays are
                accepted; channel layout is adjusted to match the model.
            pbar (bool, optional): Whether lower-level inference may display
                progress bars. Defaults to True.
            stems (str | Sequence[str] | None, optional): Stem name or stem
                names to return. ``None`` returns all model stems. Defaults to
                None.

        Returns:
            dict[str, np.ndarray]: Mapping of stem name to separated audio.

        Example:
            >>> results = separator.separate(audio, stems=["vocals", "instrumental"])
            >>> vocals = results["vocals"]

        Notes:
            When output ``normalize=True``, the shared normalization gain is
            computed only across the returned stems."""
        return self._separate(mix, pbar=pbar, stems=stems)

    def _separate(self, mix, pbar, stems=None):
        """Internal separation implementation.

        Args:
            mix (np.ndarray): Input waveform.
            pbar (bool): Whether progress bars are enabled.
            stems (str | Sequence[str] | None, optional): Stem subset to
                separate. Defaults to None.

        Returns:
            dict[str, np.ndarray]: Separated stems.

        Notes:
            This method prepares channel layout, applies legacy input
            standardization when enabled, runs TTA variants when requested,
            builds stem results, and finally applies linked output peak
            normalization when ``self.output_normalize`` is true."""
        mix = _prepare_mix_channels(mix, _model_is_stereo(self.model_type, self.config), self.logger)
        if self.model_type == "vr":
            results = self.model.separate_array(mix, self.config.audio.get("sample_rate", 44100))
            return _normalize_outputs(results, self.output_normalize, self.logger)

        instruments, source_indices = _resolve_instruments(self.config, stems)
        if self.config.training.target_instrument is not None:
            self.logger.debug(
                "Target instrument is not null, set primary_stem to target_instrument, secondary_stem will be calculated by mix - target_instrument"
            )

        mix_orig = mix.copy()
        mix, standardize_stats = _standardize_mix(mix, self.config.inference.get("normalize", False), self.logger)
        full_result = [
            demix(
                self.config,
                self.model,
                track,
                self.device,
                pbar=pbar,
                model_type=self.model_type,
                source_indices=source_indices,
                progress_callback=self.progress_callback,
            )
            for track in _tta_variants(mix, self.use_tta, self.logger)
        ]

        self.logger.debug("Finished demixing tracks.")
        waveforms = _merge_tta_results(full_result)
        self.logger.debug(f"Starting to extract waveforms for instruments: {instruments}")
        results = _build_results(waveforms, instruments, mix_orig, self.config, standardize_stats, self.logger)
        results = _normalize_outputs(results, self.output_normalize, self.logger)
        self.logger.debug("Separation process completed.")
        return results

    def save_audio(self, audio, sr, file_name, store_dir):
        """Save one audio array using the separator output settings.

        Args:
            audio (np.ndarray): Audio samples to write.
            sr (int): Sample rate in Hz.
            file_name (str): Output filename without extension. The method
                appends ``self.output_format``.
            store_dir (str | os.PathLike): Output directory.

        Returns:
            None: The encoded audio file is written to disk.

        Example:
            >>> separator.output_format = "wav"
            >>> separator.save_audio(results["vocals"], 44100, "song_vocals", "results")"""
        output_format = self.output_format.lower()
        file = os.path.join(store_dir, f"{file_name}.{output_format}")
        save_audio(file, audio, sr, output_format, self.audio_params)

    def close(self):
        """Release model references and clear backend caches.

        Args:
            None: This callable does not accept user-provided arguments.

        Returns:
            None: Model references are dropped and DirectML/CUDA/MPS/MLX caches
            are cleared where available. The ``torch.backends.cudnn.benchmark``
            flag is also restored to the value it held before the separator
            was initialized, so embedding pymss in a larger pipeline does not
            leak the benchmark-enabled side effect into other modules.

        Example:
            >>> separator.close()"""
        self.logger.debug("Closing separator and releasing model references...")
        model = getattr(self, "model", None)
        model_run = None
        try:
            if _device_type(self.device) == "privateuseone":
                # These are ordinary attributes rather than registered buffers,
                # so model.to("cpu") leaves their DirectML allocations behind.
                cleared = _clear_dml_model_caches(model, self.logger)
                if cleared:
                    self.logger.debug("Cleared %d DirectML model cache containers.", cleared)
            if self.model_type == "vr" and model is not None:
                model_run = getattr(model, "model_run", None)
                if model_run is not None and hasattr(model_run, "to"):
                    try:
                        model_run.to("cpu")
                    except Exception as exc:
                        self.logger.debug(f"Could not move VR model to CPU during close: {exc}")
                if hasattr(model, "model_run"):
                    model.model_run = None
            elif model is not None and hasattr(model, "to"):
                try:
                    model.to("cpu")
                except Exception as exc:
                    self.logger.debug(f"Could not move model to CPU during close: {exc}")
        finally:
            self._restore_cudnn_benchmark()
            self.model = None
            self.config = None
            self.store_dirs = {}
            model_run = None
            model = None
            self.del_cache()

    def _restore_cudnn_benchmark(self):
        """Restore ``torch.backends.cudnn.benchmark`` to its pre-init value.

        Args:
            None: This callable does not accept user-provided arguments.

        Returns:
            None: When the separator captured an initial ``cudnn.benchmark``
            value during initialization, the global flag is restored to that
            value so downstream modules observe the same state as before
            pymss ran."""
        initial = getattr(self, "_cudnn_benchmark_initial", None)
        if initial is None:
            return
        try:
            torch.backends.cudnn.benchmark = initial
        except Exception as exc:
            self.logger.debug(f"Could not restore torch.backends.cudnn.benchmark: {exc}")

    def del_cache(self):
        """Run garbage collection and clear accelerator memory caches.

        Args:
            None: This callable does not accept user-provided arguments.

        Returns:
            None: Python garbage collection runs, DirectML model cache references
            are released, and CUDA or MPS/MLX caches are emptied for the active
            device."""
        self.logger.debug("Running garbage collection...")
        if _device_type(self.device) == "privateuseone":
            cleared = _clear_dml_model_caches(getattr(self, "model", None), self.logger)
            if cleared:
                self.logger.debug("Cleared %d DirectML model cache containers.", cleared)
        gc.collect()
        if _device_type(self.device) == "mps":
            self.logger.debug("Clearing MPS cache...")
            torch.mps.empty_cache()
            clear_mlx_cache()
        if _device_type(self.device) == "cuda":
            self.logger.debug("Clearing CUDA cache...")
            torch.cuda.empty_cache()
