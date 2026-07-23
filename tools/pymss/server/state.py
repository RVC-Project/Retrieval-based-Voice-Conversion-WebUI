from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from ..config import load_config
from ..logger import get_separation_logger
from ..model_download import download_model
from ..model_registry import create_separator, resolve_model
from ..separator import INFERENCE_PARAM_TARGETS, PASSTHROUGH_INFERENCE_PARAMS
from .config import ServerConfig


DEFAULT_ENDPOINT = object()
FLOAT_INFERENCE_PARAMS = frozenset({"post_process_threshold", "overlap"})


VR_SUPPORTED_PARAMETERS = {
    "aggression",
    "batch_size",
    "enable_post_process",
    "enable_tta",
    "fuse_conv_bn",
    "high_end_process",
    "mps_model_backend",
    "mps_model_compute_dtype",
    "normalize",
    "post_process_threshold",
    "use_amp",
    "use_channels_last",
    "window_size",
}


class InferenceParameterError(ValueError):
    """Exception raised for unsupported inference parameters."""

    pass


class RequestLimiter:
    """Async request limiter backed by a semaphore.

    Args:
        limit (int): Limit value.
    """

    def __init__(self, limit):
        """Initialize the instance.

        Args:
            limit (int): Limit value.

        Returns:
            None: This method completes for its side effects."""
        self.limit = max(1, int(limit))
        self.active = 0
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquire value.

        Args:
            None: This callable does not accept user-provided arguments.

        Returns:
            Any: Computed result."""
        async with self.lock:
            if self.active >= self.limit:
                return False
            self.active += 1
            return True

    async def release(self):
        """Release value.

        Args:
            None: This callable does not accept user-provided arguments.

        Returns:
            None: This callable completes for its side effects."""
        async with self.lock:
            self.active = max(0, self.active - 1)


@dataclass
class LoadedModel:
    """Container for one loaded separator and its metadata."""

    separator: object
    entry: object
    resolved: dict
    requested_model: str
    model_id: str
    sample_rate: int
    instruments: tuple[str, ...]
    device: str
    inference_params: dict
    supported_parameters: dict[str, list[str]]
    audio_params: dict = field(default_factory=dict)

    def is_model_id(self, model):
        """Return whether model ID.

        Args:
            model (str): Model value.

        Returns:
            bool: True when the condition is satisfied."""
        return str(model or "") == self.model_id


@dataclass
class ServerState:
    """Mutable server state for the currently loaded model."""

    config: ServerConfig
    logger: object
    operation_lock: asyncio.Lock
    limiter: RequestLimiter
    model_lock: asyncio.Lock
    inference_lock: asyncio.Lock
    download_lock: asyncio.Lock
    loaded: LoadedModel | None = None
    model_loading: bool = False
    model_loading_target: str | None = None
    model_downloading: bool = False
    model_downloading_target: str | None = None

    def is_loaded_model(self, model):
        """Return whether loaded model.

        Args:
            model (str): Model value.

        Returns:
            bool: True when the condition is satisfied."""
        return self.loaded is not None and self.loaded.is_model_id(model)


def _section(config, section):
    """Implement the section helper.

    Args:
        config (AttrDict | dict): Loaded pymss configuration.
        section (Mapping | None): Section value.

    Returns:
        Any: Computed result."""
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(section)
    return getattr(config, section, None)


def _contains(section, key):
    """Implement the contains helper.

    Args:
        section (Mapping | None): Section value.
        key (str): Key value.

    Returns:
        Any: Computed result."""
    if section is None:
        return False
    if isinstance(section, dict):
        return key in section
    return hasattr(section, key)


def _is_parameter_supported(config, model_type, key, section_name):
    """Return whether parameter supported.

    Args:
        config (AttrDict | dict): Loaded pymss configuration.
        model_type (Any): Model type value.
        key (str): Key value.
        section_name (str): Section name value.

    Returns:
        bool: True when the condition is satisfied."""
    if model_type == "vr" and key in VR_SUPPORTED_PARAMETERS:
        return True
    if key == "mps_mlx_clear_cache" and model_type != "vr":
        return True
    # standardize is legacy input standardization backed by MSS YAML inference.normalize.
    # normalize is output peak normalization owned by runtime inference params.
    config_key = "normalize" if key == "standardize" else key
    return _contains(_section(config, section_name), config_key)


def supported_parameters(config, model_type):
    """Return inference parameters supported by a loaded model config.

    Args:
        config (AttrDict | dict): Loaded pymss configuration.
        model_type (Any): Model type value.

    Returns:
        Any: Computed result."""
    grouped: dict[str, list[str]] = {}
    for key, section_name in INFERENCE_PARAM_TARGETS.items():
        if not _is_parameter_supported(config, model_type, key, section_name):
            continue
        grouped.setdefault(section_name, []).append(key)
    return grouped


def validate_inference_params(params, config, model_type):
    """Validate user-provided inference parameters for a model.

    Args:
        params (dict | None): Inference parameter overrides.
        config (AttrDict | dict): Loaded pymss configuration.
        model_type (Any): Model type value.

    Returns:
        None: This callable completes for its side effects."""
    for key, value in params.items():
        section_name = INFERENCE_PARAM_TARGETS.get(key)
        if section_name is None:
            raise InferenceParameterError(f"Unknown inference parameter: {key}")
        if not _is_parameter_supported(config, model_type, key, section_name):
            raise InferenceParameterError(f"Inference parameter {key!r} is not supported by this model")
        if key in PASSTHROUGH_INFERENCE_PARAMS:
            continue
        try:
            float(value) if key in FLOAT_INFERENCE_PARAMS else int(value)
        except (TypeError, ValueError):
            raise InferenceParameterError(f"Inference parameter {key!r} must be numeric")


def _preload_config(resolved):
    """Implement the preload config helper.

    Args:
        resolved (Any): Resolved value.

    Returns:
        Any: Computed result."""
    model_type = resolved["model_type"]
    if model_type == "vr":
        return None
    config_path = resolved.get("config_path")
    return load_config(config_path) if config_path else None


def _resolve_existing_or_download(model, model_dir, source, endpoint):
    """Resolve existing or download.

    Args:
        model (str): Model value.
        model_dir (str | os.PathLike | None): Local model cache directory. Uses the package default when None.
        source (str): Download source name.
        endpoint (str | None): Optional custom download endpoint.

    Returns:
        Any: Resolved value."""
    try:
        return resolve_model(model, model_dir=model_dir, require_supported=True, require_exists=True)
    except FileNotFoundError:
        download_model(model, model_dir=model_dir, source=source, endpoint=endpoint)
        return resolve_model(model, model_dir=model_dir, require_supported=True, require_exists=True)


def load_model(config, model, source=None, endpoint=DEFAULT_ENDPOINT, inference_params=None):
    """Resolve and load a model into server state.

    Args:
        config (AttrDict | dict): Loaded pymss configuration.
        model (str): Model value.
        source (str, optional): Download source name. Defaults to "modelscope".
        endpoint (str | None, optional): Optional custom download endpoint. Defaults to None.
        inference_params (dict | None, optional): Inference params value. Defaults to None.

    Returns:
        Any: Computed result."""
    source = source or config.source
    endpoint = config.endpoint if endpoint is DEFAULT_ENDPOINT else endpoint
    params = dict(config.inference_params or {})
    if inference_params is not None:
        params.update(inference_params)

    resolved = _resolve_existing_or_download(model, config.model_dir, source, endpoint)
    pre_config = _preload_config(resolved)
    validate_inference_params(params, pre_config, resolved["model_type"])

    separator = create_separator(
        model,
        model_dir=config.model_dir,
        device=config.device,
        device_ids=config.device_ids or [0],
        output_format="wav",
        store_dirs="results",
        logger=get_separation_logger(),
        debug=config.debug,
        inference_params=params,
    )
    instruments = tuple(str(item) for item in separator.config.training.instruments)
    sample_rate = int(separator.config.audio.get("sample_rate", 44100))
    entry = resolved["entry"]
    model_type = getattr(separator, "model_type", resolved["model_type"])
    return LoadedModel(
        separator=separator,
        entry=entry,
        resolved=resolved,
        requested_model=model,
        model_id=entry.name,
        sample_rate=sample_rate,
        instruments=instruments,
        device=separator.device,
        inference_params=params,
        supported_parameters=supported_parameters(separator.config, model_type),
        audio_params=dict(getattr(separator, "audio_params", {}) or {}),
    )


def close_loaded_model(loaded):
    """Close and release resources held by a loaded model.

    Args:
        loaded (LoadedModel): Loaded value.

    Returns:
        None: This callable completes for its side effects."""
    if loaded is None:
        return
    separator = loaded.separator
    close = getattr(separator, "close", None)
    if close is not None:
        close()


def load_state(config):
    """Create the initial server state.

    Args:
        config (AttrDict | dict): Loaded pymss configuration.

    Returns:
        Any: Computed result."""
    logger = get_separation_logger()
    state = ServerState(
        config=config,
        logger=logger,
        operation_lock=asyncio.Lock(),
        limiter=RequestLimiter(config.max_queue_size),
        model_lock=asyncio.Lock(),
        inference_lock=asyncio.Lock(),
        download_lock=asyncio.Lock(),
    )
    if config.model:
        state.loaded = load_model(config, config.model)
    return state


def model_card(loaded):
    """Build metadata for the currently loaded model.

    Args:
        loaded (LoadedModel): Loaded value.

    Returns:
        Any: Computed result."""
    entry = loaded.entry
    return {
        "id": loaded.model_id,
        "object": "model",
        "created": 0,
        "owned_by": "pymss",
        "pymss": {
            "catalog_name": entry.name,
            "model_type": entry.model_type,
            "architecture": entry.architecture,
            "category": entry.category_path or entry.primary_category,
            "catalog_target_stem": entry.target_stem,
            "supported": entry.supported,
            "sample_rate": loaded.sample_rate,
            "instruments": list(loaded.instruments),
            "instruments_source": "separator.config.training.instruments",
            "supported_parameters": loaded.supported_parameters,
        },
    }
