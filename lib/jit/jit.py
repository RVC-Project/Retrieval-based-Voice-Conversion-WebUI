import pickle
from io import BytesIO
from collections import OrderedDict
from pathlib import Path
import os
from typing import Literal

import torch

from lib.types import FileLike


def load_pickle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(ckpt: dict, save_path: Path):
    with open(save_path, "wb") as f:
        pickle.dump(ckpt, f)


def load_inputs(path: FileLike, device: str, is_half=False):  # type: ignore
    parm = torch.load(path, map_location=torch.device("cpu"))
    for key in parm.keys():
        parm[key] = parm[key].to(device)
        if is_half and parm[key].dtype == torch.float32:
            parm[key] = parm[key].half()
        elif not is_half and parm[key].dtype == torch.float16:
            parm[key] = parm[key].float()
    return parm


def export_jit_model(
    model: torch.nn.Module,
    mode: Literal["trace", "script"] = "trace",
    inputs: dict = None,
    device=torch.device("cpu"),
    is_half: bool = False,
) -> dict:
    model = model.half() if is_half else model.float()
    model.eval()
    if mode == "trace":
        assert inputs is not None
        model_jit = torch.jit.trace(model, example_kwarg_inputs=inputs)
    elif mode == "script":
        model_jit = torch.jit.script(model)
    model_jit.to(device)
    model_jit = model_jit.half() if is_half else model_jit.float()
    buffer = BytesIO()
    # model_jit=model_jit.cpu()
    torch.jit.save(model_jit, buffer)
    del model_jit
    cpt = OrderedDict()
    cpt["model"] = buffer.getvalue()
    cpt["is_half"] = is_half
    return cpt


def get_jit_model(model_path: Path, is_half: bool, device: str, exporter):
    stem = model_path.with_suffix("")
    jit_model_path = stem.with_suffix(".half.jit" if is_half else ".jit")
    ckpt = None

    if jit_model_path.exists():
        ckpt = load_pickle(jit_model_path)
        model_device = ckpt["device"]
        if model_device != str(device):
            del ckpt
            ckpt = None

    if ckpt is None:
        ckpt = exporter(
            model_path=model_path,
            mode="script",
            inputs_path=None,
            save_path=jit_model_path,
            device=device,
            is_half=is_half,
        )

    return ckpt
