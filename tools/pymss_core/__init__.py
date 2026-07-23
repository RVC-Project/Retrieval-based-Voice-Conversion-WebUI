"""Core model, configuration, and checkpoint API for music source separation.

`pymss_core` contains the shared pieces used by higher-level packages:
configuration loading, model construction, model definitions, and
checkpoint/state-dict helpers. It intentionally does not provide file audio
I/O, inference DSP pipelines, chunked demixing, model catalog downloads, CLI,
HTTP server, or WebUI functionality.
"""

from .checkpoint import load_checkpoint, load_model_weights, load_state_dict, unwrap_state_dict
from .config import AttrDict, ConfigLoader, load_config, to_attrdict, to_plain
from .utils import get_model_from_config

__all__ = (
    "AttrDict",
    "ConfigLoader",
    "get_model_from_config",
    "load_checkpoint",
    "load_config",
    "load_model_weights",
    "load_state_dict",
    "to_attrdict",
    "to_plain",
    "unwrap_state_dict",
)
