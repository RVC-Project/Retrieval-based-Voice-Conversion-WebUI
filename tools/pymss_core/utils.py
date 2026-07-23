"""Model construction helpers."""

from __future__ import annotations

from .config import load_config


def get_model_from_config(model_type, config_path, model_kwargs_override=None):
    """Instantiate a separation model from a model configuration file."""
    model_kwargs_override = model_kwargs_override or {}
    config = load_config(config_path)

    if model_type == "mdx23c":
        from .modules.mdx23c_tfc_tdf_v3 import TFC_TDF_net

        return TFC_TDF_net(config), config
    if model_type == "htdemucs":
        from .modules.demucs4ht import get_model

        return get_model(config), config
    if model_type == "mel_band_roformer":
        from .modules.bs_roformer import MelBandRoformer

        model_kwargs = dict(config.model)
        model_kwargs.update(model_kwargs_override)
        return MelBandRoformer(**model_kwargs), config
    if model_type == "bs_roformer":
        from .modules.bs_roformer import BSRoformer

        return BSRoformer(**dict(config.model)), config
    if model_type == "bs_roformer_hyperace":
        from .modules.bs_roformer import BSRoformerHyperACE

        return BSRoformerHyperACE(**dict(config.model)), config
    if model_type == "bandit":
        from .modules.bandit.core.model import MultiMaskMultiSourceBandSplitRNNSimple

        return MultiMaskMultiSourceBandSplitRNNSimple(**config.model), config
    if model_type == "bandit_v2":
        from .modules.bandit_v2.bandit import Bandit

        return Bandit(**config.kwargs), config
    if model_type == "scnet":
        from .modules.scnet import SCNet

        return SCNet(**config.model), config
    if model_type == "apollo":
        from .modules.look2hear.apollo import Apollo

        return Apollo(**config.model), config
    if model_type == "vr":
        raise ValueError("VR network modules do not use YAML config loading")
    raise ValueError(f"Model type {model_type} not supported")
