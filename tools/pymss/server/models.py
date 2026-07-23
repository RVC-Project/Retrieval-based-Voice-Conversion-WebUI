from __future__ import annotations

from ..model_download import remote_url
from ..model_registry import (
    auxiliary_paths_for,
    config_path_for,
    get_model_entry,
    list_models,
    model_path_for,
    model_root,
)


def _bool_query(value, *, default=False):
    """Implement the bool query helper.

    Args:
        value (Any): Value value.
        default (Any, optional): Default value. Defaults to False.

    Returns:
        Any: Computed result."""
    if value is None:
        return default
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError("Expected boolean value")


def parse_supported_filter(value):
    """Parse the model catalog supported filter.

    Args:
        value (Any): Value value.

    Returns:
        Any: Parsed value."""
    if value is None:
        return True
    value = str(value).strip().lower()
    if value == "all":
        return None
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError("supported must be true, false, or all")


def parse_local_filter(value):
    """Parse the model catalog local-file filter.

    Args:
        value (Any): Value value.

    Returns:
        Any: Parsed value."""
    if value is None:
        return "all"
    value = str(value).strip().lower()
    if value not in {"all", "complete", "missing"}:
        raise ValueError("local must be all, complete, or missing")
    return value


def parse_include_files(value):
    """Parse whether model file metadata should be included.

    Args:
        value (Any): Value value.

    Returns:
        Any: Parsed value."""
    return _bool_query(value, default=False)


def _entry_file_specs(entry, model_dir=None):
    """Implement the entry file specs helper.

    Args:
        entry (ModelEntry): Entry value.
        model_dir (str | os.PathLike | None, optional): Local model cache directory. Uses the package default when None. Defaults to None.

    Returns:
        Any: Computed result."""
    specs = [("model", entry.relpath, model_path_for(entry, model_dir))]
    config_path = config_path_for(entry, model_dir)
    if entry.config_relpath and config_path is not None:
        specs.append(("config", entry.config_relpath, config_path))
    specs.extend(
        ("auxiliary", relpath, path) for relpath, path in zip(entry.auxiliary_relpaths, auxiliary_paths_for(entry, model_dir))
    )
    return specs


def local_file_status(entry, model_dir=None):
    """Report whether a catalog model exists in the local model directory.

    Args:
        entry (ModelEntry): Entry value.
        model_dir (str | os.PathLike | None, optional): Local model cache directory. Uses the package default when None. Defaults to None.

    Returns:
        Any: Computed result."""
    specs = _entry_file_specs(entry, model_dir)
    missing = [relpath for _role, relpath, path in specs if not path.is_file()]
    return {
        "complete": not missing,
        "missing_count": len(missing),
    }


def catalog_model_files(entry, model_dir=None, source="modelscope", endpoint=None):
    """Build file metadata for a catalog model.

    Args:
        entry (ModelEntry): Entry value.
        model_dir (str | os.PathLike | None, optional): Local model cache directory. Uses the package default when None. Defaults to None.
        source (str, optional): Download source name. Defaults to "modelscope".
        endpoint (str | None, optional): Optional custom download endpoint. Defaults to None.

    Returns:
        Any: Computed result."""
    files = []
    for role, relpath, path in _entry_file_specs(entry, model_dir):
        exists = path.is_file()
        try:
            size_bytes = path.stat().st_size if exists else 0
        except OSError:
            exists = False
            size_bytes = 0
        files.append(
            {
                "role": role,
                "relpath": relpath,
                "exists": exists,
                "size_bytes": size_bytes,
                "remote_url": remote_url(relpath, source=source, endpoint=endpoint),
            }
        )
    return files


def catalog_model_card(entry, model_dir=None, source="modelscope", endpoint=None, include_files=False):
    """Build a summary card for a catalog model.

    Args:
        entry (ModelEntry): Entry value.
        model_dir (str | os.PathLike | None, optional): Local model cache directory. Uses the package default when None. Defaults to None.
        source (str, optional): Download source name. Defaults to "modelscope".
        endpoint (str | None, optional): Optional custom download endpoint. Defaults to None.
        include_files (Any, optional): Include files value. Defaults to False.

    Returns:
        Any: Computed result."""
    category = entry.category_path or entry.primary_category
    pymss = {
        "name": entry.name,
        "aliases": list(entry.aliases),
        "model_type": entry.model_type,
        "architecture": entry.architecture,
        "category": category,
        "primary_category": entry.primary_category,
        "secondary_category": entry.secondary_category,
        "target_stem": entry.target_stem,
        "supported": entry.supported,
        "unsupported_reason": entry.unsupported_reason,
        "size_bytes": entry.size_bytes,
        "local": local_file_status(entry, model_dir),
        "remote": {
            "available": True,
            "source": source,
            "endpoint": endpoint,
        },
    }
    if include_files:
        pymss["files"] = catalog_model_files(entry, model_dir, source=source, endpoint=endpoint)
    return {
        "id": entry.name,
        "object": "pymss.model_catalog_entry",
        "owned_by": "pymss",
        "pymss": pymss,
    }


def catalog_model_detail(model, model_dir=None, source="modelscope", endpoint=None):
    """Build the detailed response for one catalog model.

    Args:
        model (str): Model value.
        model_dir (str | os.PathLike | None, optional): Local model cache directory. Uses the package default when None. Defaults to None.
        source (str, optional): Download source name. Defaults to "modelscope".
        endpoint (str | None, optional): Optional custom download endpoint. Defaults to None.

    Returns:
        Any: Computed result."""
    entry = get_model_entry(model)
    return catalog_model_card(entry, model_dir=model_dir, source=source, endpoint=endpoint, include_files=True)


def filter_catalog_models(category=None, supported=True, local="all", q=None, model_dir=None):
    """Filter catalog models by category, support, local files, and text query.

    Args:
        category (Any, optional): Category value. Defaults to None.
        supported (bool | None, optional): Optional support-status filter. Defaults to None.
        local (Any, optional): Local value. Defaults to 'all'.
        q (Any, optional): Q value. Defaults to None.
        model_dir (str | os.PathLike | None, optional): Local model cache directory. Uses the package default when None. Defaults to None.

    Returns:
        Any: Computed result."""
    rows = list_models(category=category, supported=supported)
    query = str(q or "").strip().lower()
    if query:
        rows = [
            entry
            for entry in rows
            if query in entry.name.lower()
            or any(query in alias.lower() for alias in entry.aliases)
            or query in (entry.architecture or "").lower()
            or query in (entry.target_stem or "").lower()
        ]
    if local != "all":
        want_complete = local == "complete"
        rows = [entry for entry in rows if local_file_status(entry, model_dir)["complete"] is want_complete]
    return rows
