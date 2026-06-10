"""Construct all synthesizers + discriminators from real configs and print a
state_dict fingerprint.

Run before and after the models.py split; output must be identical:
    .venv/bin/python tests/model_construct.py
"""

import hashlib
import json
import os
import sys

sys.path.insert(0, os.getcwd())

import torch

from infer.lib.infer_pack.models import (
    MultiPeriodDiscriminator,
    MultiPeriodDiscriminatorV2,
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)

torch.manual_seed(0)


def fingerprint(model):
    keys = sorted(model.state_dict().keys())
    h = hashlib.sha256("\n".join(keys).encode()).hexdigest()[:16]
    return len(keys), h


def build_args(cfg):
    return (
        cfg["data"]["filter_length"] // 2 + 1,
        cfg["train"]["segment_size"] // cfg["data"]["hop_length"],
    )


with open("configs/v1/40k.json") as f:
    cfg = json.load(f)
args = build_args(cfg)
print(
    "SynthesizerTrnMs256NSFsid",
    *fingerprint(
        SynthesizerTrnMs256NSFsid(
            *args, **cfg["model"], is_half=False, sr=cfg["data"]["sampling_rate"]
        )
    ),
)
print(
    "SynthesizerTrnMs256NSFsid_nono",
    *fingerprint(SynthesizerTrnMs256NSFsid_nono(*args, **cfg["model"], is_half=False)),
)

with open("configs/v2/48k.json") as f:
    cfg = json.load(f)
args = build_args(cfg)
print(
    "SynthesizerTrnMs768NSFsid",
    *fingerprint(
        SynthesizerTrnMs768NSFsid(
            *args, **cfg["model"], is_half=False, sr=cfg["data"]["sampling_rate"]
        )
    ),
)
print(
    "SynthesizerTrnMs768NSFsid_nono",
    *fingerprint(SynthesizerTrnMs768NSFsid_nono(*args, **cfg["model"], is_half=False)),
)

print("MPD", *fingerprint(MultiPeriodDiscriminator(False)))
print("MPDv2", *fingerprint(MultiPeriodDiscriminatorV2(False)))
print("DONE")
