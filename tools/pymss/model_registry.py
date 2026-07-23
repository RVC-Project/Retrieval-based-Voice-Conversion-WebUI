import json
import os
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path


def _default_model_dir():
    """Implement the default model dir helper.

    Args:
        None: This callable does not accept user-provided arguments.

    Returns:
        Any: Computed result."""
    env_value = os.environ.get("PYMSS_MODEL_DIR")
    if env_value:
        return Path(env_value)
    repo_models = Path(__file__).resolve().parent.parent / "all_models"
    if repo_models.is_dir():
        return repo_models
    return Path.home() / ".cache" / "pymss" / "models"


DEFAULT_MODEL_DIR = _default_model_dir()


@dataclass(frozen=True)
class ModelEntry:
    """Catalog metadata for one downloadable pymss model."""

    name: str
    aliases: tuple
    model_type: str | None
    architecture: str
    supported: bool
    unsupported_reason: str
    relpath: str
    config_relpath: str
    auxiliary_relpaths: tuple
    size_bytes: int
    sha256: str
    primary_category: str
    primary_category_cn: str
    secondary_category: str
    secondary_category_cn: str
    target_stem: str
    config_instruments: str
    config_target_instrument: str
    classification_confidence: str
    classification_basis: str

    @property
    def stem(self):
        """Implement the stem helper.

        Args:
            None: This callable does not accept user-provided arguments.

        Returns:
            Any: Computed result."""
        return Path(self.name).stem

    @property
    def category_path(self):
        """Implement the category path helper.

        Args:
            None: This callable does not accept user-provided arguments.

        Returns:
            Any: Computed result."""
        return "/".join(part for part in (self.primary_category, self.secondary_category) if part)

    @classmethod
    def from_dict(cls, data):
        """Implement the from dict helper.

        Args:
            data (Mapping | None): Data value.

        Returns:
            Any: Computed result."""
        return cls(
            name=data["name"],
            aliases=tuple(data.get("aliases", ())),
            model_type=data.get("model_type"),
            architecture=data.get("architecture", ""),
            supported=bool(data.get("supported", False)),
            unsupported_reason=data.get("unsupported_reason", ""),
            relpath=data["relpath"],
            config_relpath=data.get("config_relpath", ""),
            auxiliary_relpaths=tuple(data.get("auxiliary_relpaths", ())),
            size_bytes=int(data.get("size_bytes", 0)),
            sha256=data.get("sha256", ""),
            primary_category=data.get("primary_category", ""),
            primary_category_cn=data.get("primary_category_cn", ""),
            secondary_category=data.get("secondary_category", ""),
            secondary_category_cn=data.get("secondary_category_cn", ""),
            target_stem=data.get("target_stem", ""),
            config_instruments=data.get("config_instruments", ""),
            config_target_instrument=data.get("config_target_instrument", ""),
            classification_confidence=data.get("classification_confidence", ""),
            classification_basis=data.get("classification_basis", ""),
        )


@lru_cache(maxsize=1)
def load_model_catalog():
    """Load model catalog.

    Args:
        None: This callable does not accept user-provided arguments.

    Returns:
        Any: Computed result."""
    with resources.files("pymss.resources").joinpath("model_catalog.json").open(encoding="utf-8") as f:
        data = json.load(f)
    models = [ModelEntry.from_dict(item) for item in data["models"]]
    return {**data, "models": models}


@lru_cache(maxsize=1)
def _model_index():
    """Implement the model index helper.

    Args:
        None: This callable does not accept user-provided arguments.

    Returns:
        Any: Computed result."""
    index = {}
    for entry in load_model_catalog()["models"]:
        names = {entry.name, entry.stem, *entry.aliases}
        for name in names:
            key = _normalize_model_name(name)
            if key in index and index[key].name != entry.name:
                continue
            index[key] = entry
    return index


def _normalize_model_name(name):
    """Normalize model name.

    Args:
        name (Any): Name value.

    Returns:
        Any: Computed result."""
    return str(name).strip().lower()


def list_models(category=None, supported=None):
    """List model catalog entries.

    The catalog contains every model known to pymss, including unsupported
    entries. Use the filters when building model selectors, download tools, or
    validation code.

    Args:
        category (str | None, optional): Optional category filter. The value is
            matched against primary category, secondary category, or combined
            ``primary/secondary`` category path. Matching is case-insensitive.
            Defaults to None.
        supported (bool | None, optional): Support-status filter. ``True``
            returns only models supported by the current inference code,
            ``False`` returns unsupported entries, and ``None`` returns all
            catalog entries. Defaults to None.

    Returns:
        list[ModelEntry]: Matching catalog entries in catalog order.

    Example:
        >>> from pymss import list_models
        >>> supported_models = list_models(supported=True)
        >>> supported_models[0].name

    Example:
        >>> vocal_models = list_models(category="vocal", supported=True)
        >>> [model.stem for model in vocal_models[:3]]"""
    models = load_model_catalog()["models"]
    if category:
        category = category.lower()
        models = [
            item
            for item in models
            if item.primary_category.lower() == category
            or item.secondary_category.lower() == category
            or item.category_path.lower() == category
        ]
    if supported is not None:
        models = [item for item in models if item.supported is bool(supported)]
    return models


def get_model_entry(model_name):
    """Return catalog metadata for one model name or alias.

    Args:
        model_name (str): Full catalog filename, stem name, or alias. Matching
            is case-insensitive after stripping surrounding whitespace.

    Returns:
        ModelEntry: Catalog entry containing architecture, support status,
        relative file paths, hashes, categories, target stem, and aliases.

    Raises:
        KeyError: If ``model_name`` is unknown.

    Example:
        >>> from pymss import get_model_entry
        >>> entry = get_model_entry("bs_roformer_voc_hyperacev2")
        >>> entry.model_type
        'bs_roformer'

    Example:
        >>> entry.supported, entry.category_path
        (True, entry.category_path)"""
    try:
        return _model_index()[_normalize_model_name(model_name)]
    except KeyError as exc:
        raise KeyError(f"Unknown pymss model: {model_name}") from exc


def model_root(model_dir=None):
    """Implement the model root helper.

    Args:
        model_dir (str | os.PathLike | None, optional): Local model cache directory. Uses the package default when None. Defaults to None.

    Returns:
        Any: Computed result."""
    return Path(model_dir).expanduser() if model_dir else DEFAULT_MODEL_DIR


def model_path_for(entry, model_dir=None):
    """Implement the model path for helper.

    Args:
        entry (ModelEntry): Entry value.
        model_dir (str | os.PathLike | None, optional): Local model cache directory. Uses the package default when None. Defaults to None.

    Returns:
        Any: Computed result."""
    return model_root(model_dir) / entry.relpath


def config_path_for(entry, model_dir=None):
    """Implement the config path for helper.

    Args:
        entry (ModelEntry): Entry value.
        model_dir (str | os.PathLike | None, optional): Local model cache directory. Uses the package default when None. Defaults to None.

    Returns:
        Any: Computed result."""
    return model_root(model_dir) / entry.config_relpath if entry.config_relpath else None


def auxiliary_paths_for(entry, model_dir=None):
    """Implement the auxiliary paths for helper.

    Args:
        entry (ModelEntry): Entry value.
        model_dir (str | os.PathLike | None, optional): Local model cache directory. Uses the package default when None. Defaults to None.

    Returns:
        Any: Computed result."""
    root = model_root(model_dir)
    return [root / relpath for relpath in entry.auxiliary_relpaths]


def resolve_model(model_name, model_dir=None, require_supported=True, require_exists=True):
    """Resolve a catalog model to local file paths.

    This function does not instantiate a model. It only translates a catalog
    name or alias into the local weights/config paths that ``MSSeparator`` will
    use.

    Args:
        model_name (str): Model name, stem, or alias from the pymss catalog.
        model_dir (str | os.PathLike | None, optional): Local model cache
            directory. When omitted, pymss uses ``PYMSS_MODEL_DIR`` if set, a
            repository-local ``all_models`` directory if present, or the user
            cache under ``~/.cache/pymss/models``. Defaults to None.
        require_supported (bool, optional): Whether unsupported catalog entries
            should raise ``ValueError``. Defaults to True.
        require_exists (bool, optional): Whether resolved model, config, and
            auxiliary files must already exist locally. Defaults to True.

    Returns:
        dict: Dictionary with ``entry`` (``ModelEntry``), ``model_type``,
        ``model_path``, and ``config_path`` keys.

    Raises:
        KeyError: If the model name is unknown.
        ValueError: If the model is unsupported and ``require_supported`` is
            true.
        FileNotFoundError: If required local files are missing and
            ``require_exists`` is true.

    Example:
        >>> from pymss import resolve_model
        >>> resolved = resolve_model("bs_roformer_voc_hyperacev2", require_exists=False)
        >>> resolved["model_type"]
        'bs_roformer'

    Example:
        >>> resolved = resolve_model("bs_roformer_voc_hyperacev2", model_dir="models")
        >>> resolved["model_path"].endswith(".ckpt") or resolved["model_path"].endswith(".pth")
        True"""
    entry = get_model_entry(model_name)
    if require_supported and not entry.supported:
        reason = entry.unsupported_reason or "unsupported"
        raise ValueError(f"Model {entry.name} cannot be used for inference yet: {reason}")

    model_path = model_path_for(entry, model_dir)
    config_path = config_path_for(entry, model_dir)
    missing = []
    if require_exists and not model_path.is_file():
        missing.append(str(model_path))
    if require_exists and config_path is not None and not config_path.is_file():
        missing.append(str(config_path))
    for path in auxiliary_paths_for(entry, model_dir):
        if require_exists and not path.is_file():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing model file(s): " + ", ".join(missing))

    return {
        "entry": entry,
        "model_type": entry.model_type,
        "model_path": str(model_path),
        "config_path": str(config_path) if config_path else None,
    }


def create_separator(model_name, model_dir=None, **separator_kwargs):
    """Create ``MSSeparator`` from a catalog model name.

    This is a convenience wrapper around ``resolve_model(...)`` followed by
    ``MSSeparator(...)``. It expects the model files to already exist locally;
    call ``download_model(...)`` first or use ``MSSeparator.from_model_name`` if
    you want optional downloading in one step.

    Args:
        model_name (str): Model name, stem, or alias from the pymss catalog.
        model_dir (str | os.PathLike | None, optional): Local model cache
            directory. Defaults to None.
        **separator_kwargs: Keyword arguments forwarded to ``MSSeparator``,
            such as ``device``, ``device_ids``, ``output_format``,
            ``store_dirs``, ``save_as_folder``, ``audio_params``, ``logger``,
            ``debug``, ``progress_callback``, and ``inference_params``.

    Returns:
        MSSeparator: Loaded separator instance ready for inference.

    Raises:
        FileNotFoundError: If required model files are not present locally.

    Example:
        >>> from pymss import create_separator
        >>> separator = create_separator(
        ...     "bs_roformer_voc_hyperacev2",
        ...     model_dir="models",
        ...     output_format="wav",
        ...     inference_params={"normalize": True},
        ... )
        >>> separator.process_folder("song.wav")

    Example:
        >>> separator = create_separator(
        ...     "some_six_stem_model",
        ...     store_dirs={"vocals": "out/vocals", "drums": "out/drums"},
        ... )"""
    from .separator import MSSeparator

    resolved = resolve_model(model_name, model_dir=model_dir, require_supported=True, require_exists=True)
    return MSSeparator(
        model_type=resolved["model_type"],
        model_path=resolved["model_path"],
        config_path=resolved["config_path"],
        **separator_kwargs,
    )
