import logging
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn
from transformers import AutoFeatureExtractor, HubertModel

from tools.cuda_graph import run_cuda_graph


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

HUBERT_MODEL_PATH = (PROJECT_ROOT / "assets" / "hubert_base").resolve()


def _device_type(device):
    if isinstance(device, torch.device):
        return device.type
    return str(device).split(":", 1)[0]


def load_hubert_model(device, is_half=False):
    """Load the local Transformers HuBERT/ContentVec model for RVC."""
    if not (HUBERT_MODEL_PATH / "config.json").is_file():
        raise FileNotFoundError(
            f"Transformers HuBERT model not found: {HUBERT_MODEL_PATH}"
        )

    dtype = torch.float16 if is_half else torch.float32
    load_options = {
        "local_files_only": True,
        "torch_dtype": dtype,
    }
    # DirectML does not implement every SDPA kernel used by Transformers.
    if _device_type(device) == "privateuseone":
        load_options["attn_implementation"] = "eager"

    logger.info(
        "Loading Transformers HuBERT from %s (%s on %s)",
        HUBERT_MODEL_PATH,
        dtype,
        device,
    )
    model = HubertModelWithFinalProj.from_pretrained(
        str(HUBERT_MODEL_PATH), **load_options
    )
    model = model.to(device)
    return model.eval()


@lru_cache(maxsize=1)
def hubert_audio_requires_normalization():
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        str(HUBERT_MODEL_PATH), local_files_only=True
    )
    return bool(feature_extractor.do_normalize)


def extract_hubert_features(model, source, version, padding_mask=None):
    """Return the RVC v1 (256-D) or v2 (768-D) HuBERT representation.

    Transformers hidden_states[N] is numerically equivalent to the source checkpoint's
    output_layer=N for this converted checkpoint. RVC v1 uses layer 9 followed
    by final_proj; RVC v2 uses the final (12th) encoder layer directly.
    """
    if version not in {"v1", "v2"}:
        raise ValueError(f"Unsupported RVC feature version: {version!r}")

    attention_mask = None
    if padding_mask is not None and bool(torch.any(padding_mask).item()):
        attention_mask = (~padding_mask.bool()).long()

    if version == "v1":
        if attention_mask is None:
            def forward(input_values):
                outputs = model(
                    input_values=input_values,
                    attention_mask=None,
                    output_hidden_states=True,
                    return_dict=True,
                )
                return model.final_proj(outputs.hidden_states[9])

            return run_cuda_graph(model, "hubert-v1-no-mask", forward, source)

        def forward(input_values, mask):
            outputs = model(
                input_values=input_values,
                attention_mask=mask,
                output_hidden_states=True,
                return_dict=True,
            )
            return model.final_proj(outputs.hidden_states[9])

        return run_cuda_graph(
            model, "hubert-v1-mask", forward, source, attention_mask
        )

    if attention_mask is None:
        def forward(input_values):
            return model(
                input_values=input_values,
                attention_mask=None,
                output_hidden_states=False,
                return_dict=True,
            ).last_hidden_state

        return run_cuda_graph(model, "hubert-v2-no-mask", forward, source)

    def forward(input_values, mask):
        return model(
            input_values=input_values,
            attention_mask=mask,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state

    return run_cuda_graph(model, "hubert-v2-mask", forward, source, attention_mask)
