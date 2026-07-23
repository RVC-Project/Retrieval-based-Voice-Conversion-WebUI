"""Checkpoint helpers shared by inference and training frontends."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

import torch


STATE_DICT_KEYS = ("state", "state_dict", "model_state_dict")


def unwrap_state_dict(checkpoint: Any) -> Any:
    """Return the model state dict from common MSS checkpoint containers."""
    if isinstance(checkpoint, dict):
        for key in STATE_DICT_KEYS:
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def _install_demucs_pickle_stubs() -> dict[str, ModuleType | None]:
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


def _restore_modules(previous: dict[str, ModuleType | None]) -> None:
    import sys

    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _torch_load(path: str | Path, *, map_location="cpu", weights_only: bool | None = None, mmap: bool = True) -> Any:
    kwargs: dict[str, Any] = {"map_location": map_location}
    if weights_only is not None:
        kwargs["weights_only"] = weights_only
    if mmap:
        kwargs["mmap"] = True
    try:
        return torch.load(path, **kwargs)
    except TypeError:
        kwargs.pop("mmap", None)
        try:
            return torch.load(path, **kwargs)
        except TypeError:
            kwargs.pop("weights_only", None)
            return torch.load(path, **kwargs)


def load_checkpoint(
    path: str | Path,
    *,
    model_type: str | None = None,
    map_location: str | torch.device = "cpu",
    weights_only: bool | None = None,
    mmap: bool = True,
) -> Any:
    """Load a checkpoint package with compatibility for common MSS formats."""
    model_type = (model_type or "").lower()
    if model_type in {"htdemucs", "demucs", "legacy_demucs", "legacy_tasnet"}:
        previous = _install_demucs_pickle_stubs()
        try:
            return _torch_load(path, map_location=map_location, weights_only=False, mmap=mmap)
        finally:
            _restore_modules(previous)
    if model_type == "apollo":
        weights_only = False if weights_only is None else weights_only
    return _torch_load(path, map_location=map_location, weights_only=weights_only, mmap=mmap)


def load_state_dict(
    path: str | Path,
    *,
    model_type: str | None = None,
    map_location: str | torch.device = "cpu",
    weights_only: bool | None = None,
    mmap: bool = True,
) -> Any:
    """Load and unwrap the model state dict from a checkpoint file."""
    return unwrap_state_dict(
        load_checkpoint(
            path,
            model_type=model_type,
            map_location=map_location,
            weights_only=weights_only,
            mmap=mmap,
        )
    )


def load_model_weights(
    model: torch.nn.Module,
    checkpoint_or_path: Any,
    *,
    model_type: str | None = None,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> Any:
    """Load weights from a checkpoint package or file into a model."""
    if isinstance(checkpoint_or_path, (str, Path)):
        state_dict = load_state_dict(checkpoint_or_path, model_type=model_type, map_location=map_location)
    else:
        state_dict = unwrap_state_dict(checkpoint_or_path)
    return model.load_state_dict(state_dict, strict=strict)
