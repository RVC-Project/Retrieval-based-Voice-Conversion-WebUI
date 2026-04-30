from collections import OrderedDict
from typing import Literal

import torch

from .layers.synthesizers import SynthesizerTrnMsNSFsid
from .jit import load_inputs, export_jit_model, save_pickle
from .types import FileLike


def get_synthesizer(cpt: OrderedDict, device=torch.device("cpu")):
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        encoder_dim = 256
    elif version == "v2":
        encoder_dim = 768
    net_g = SynthesizerTrnMsNSFsid(
        *cpt["config"],
        encoder_dim=encoder_dim,
        use_f0=if_f0 == 1,
    )
    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g = net_g.float()
    net_g.eval().to(device)
    net_g.remove_weight_norm()
    return net_g, cpt


def load_synthesizer(pth_path: FileLike, device=torch.device("cpu")):  # type: ignore
    return get_synthesizer(
        torch.load(pth_path, map_location=torch.device("cpu"), weights_only=True),
        device,
    )


def synthesizer_jit_export(
    model_path: str,
    mode: Literal["script", "trace"] = "script",
    inputs_path: str = None,
    save_path: str = None,
    device=torch.device("cpu"),
    is_half=False,
):
    if not save_path:
        save_path = model_path.rstrip(".pth")
        save_path += ".half.jit" if is_half else ".jit"
    if "cuda" in str(device) and ":" not in str(device):
        device = torch.device("cuda:0")
    from rvc.synthesizer import load_synthesizer

    model, cpt = load_synthesizer(model_path, device)
    assert isinstance(cpt, dict)
    model.forward = model.infer
    inputs = None
    if mode == "trace":
        inputs = load_inputs(inputs_path, device, is_half)
    ckpt = export_jit_model(model, mode, inputs, device, is_half)
    cpt.pop("weight")
    cpt["model"] = ckpt["model"]
    cpt["device"] = device
    save_pickle(cpt, save_path)
    return cpt
