from io import BytesIO
import pickle
import time
from typing import Literal, TypeAlias, cast
import torch
from tqdm import tqdm
from collections import OrderedDict

DeviceLike: TypeAlias = torch.device | str
TensorInputs: TypeAlias = dict[str, torch.Tensor]
JitMode: TypeAlias = Literal["trace", "script"]


def load_inputs(path: str, device: DeviceLike, is_half: bool = False) -> TensorInputs:
    parm = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    for key in parm.keys():
        parm[key] = parm[key].to(device)
        if is_half and parm[key].dtype == torch.float32:
            parm[key] = parm[key].half()
        elif not is_half and parm[key].dtype == torch.float16:
            parm[key] = parm[key].float()
    return parm


def benchmark(
    model: torch.nn.Module,
    inputs_path: str,
    device: DeviceLike = torch.device("cpu"),
    epoch: int = 1000,
    is_half: bool = False,
) -> None:
    parm = load_inputs(inputs_path, device, is_half)
    total_ts = 0.0
    bar = tqdm(range(epoch))
    for i in bar:
        start_time = time.perf_counter()
        o = model(**parm)
        total_ts += time.perf_counter() - start_time
    print(f"num_epoch: {epoch} | avg time(ms): {(total_ts*1000)/epoch}")


def jit_warm_up(
    model: torch.nn.Module,
    inputs_path: str,
    device: DeviceLike = torch.device("cpu"),
    epoch: int = 5,
    is_half: bool = False,
) -> None:
    benchmark(model, inputs_path, device, epoch=epoch, is_half=is_half)


def to_jit_model(
    model_path: str,
    model_type: str,
    mode: JitMode = "trace",
    inputs_path: str | None = None,
    device: DeviceLike = torch.device("cpu"),
    is_half: bool = False,
) -> tuple[torch.nn.Module, torch.jit.ScriptModule]:
    model = None
    if model_type.lower() == "synthesizer":
        from .get_synthesizer import get_synthesizer

        model, _ = get_synthesizer(model_path, device)
        model.forward = model.infer
    elif model_type.lower() == "rmvpe":
        from .get_rmvpe import get_rmvpe

        model = get_rmvpe(model_path, device)
    elif model_type.lower() == "hubert":
        from .get_hubert import get_hubert_model

        model = get_hubert_model(model_path, device)
        model.forward = model.infer
    else:
        raise ValueError(f"No model type named {model_type}")
    model = model.eval()
    model = model.half() if is_half else model.float()
    if mode == "trace":
        assert inputs_path is not None
        inputs = load_inputs(inputs_path, device, is_half)
        model_jit = cast(
            torch.jit.ScriptModule,
            torch.jit.trace(model, example_kwarg_inputs=inputs),
        )
    elif mode == "script":
        model_jit = cast(torch.jit.ScriptModule, torch.jit.script(model))
    else:
        raise ValueError(f"Unsupported JIT mode: {mode}")
    model_jit.to(device)
    model_jit = model_jit.half() if is_half else model_jit.float()
    # model = model.half() if is_half else model.float()
    return (model, model_jit)


def export(
    model: torch.nn.Module,
    mode: JitMode = "trace",
    inputs: TensorInputs | None = None,
    device: DeviceLike = torch.device("cpu"),
    is_half: bool = False,
) -> dict:
    model = model.half() if is_half else model.float()
    model.eval()
    if mode == "trace":
        assert inputs is not None
        model_jit = cast(
            torch.jit.ScriptModule,
            torch.jit.trace(model, example_kwarg_inputs=inputs),
        )
    elif mode == "script":
        model_jit = cast(torch.jit.ScriptModule, torch.jit.script(model))
    else:
        raise ValueError(f"Unsupported JIT mode: {mode}")
    model_jit.to(device)
    model_jit = model_jit.half() if is_half else model_jit.float()
    buffer = BytesIO()
    # model_jit=model_jit.cpu()
    torch.jit.save(model_jit, buffer)
    del model_jit
    cpt: OrderedDict[str, object] = OrderedDict()
    cpt["model"] = buffer.getvalue()
    cpt["is_half"] = is_half
    return cpt


def load(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def save(ckpt: dict, save_path: str) -> None:
    with open(save_path, "wb") as f:
        pickle.dump(ckpt, f)


def rmvpe_jit_export(
    model_path: str,
    mode: JitMode = "script",
    inputs_path: str | None = None,
    save_path: str | None = None,
    device: DeviceLike = torch.device("cpu"),
    is_half: bool = False,
) -> dict:
    if not save_path:
        save_path = model_path.rstrip(".pth")
        save_path += ".half.jit" if is_half else ".jit"
    if "cuda" in str(device) and ":" not in str(device):
        device = torch.device("cuda:0")
    from .get_rmvpe import get_rmvpe

    model = get_rmvpe(model_path, device)
    inputs: TensorInputs | None = None
    if mode == "trace":
        assert inputs_path is not None
        inputs = load_inputs(inputs_path, device, is_half)
    ckpt = export(model, mode, inputs, device, is_half)
    ckpt["device"] = str(device)
    save(ckpt, save_path)
    return ckpt


def synthesizer_jit_export(
    model_path: str,
    mode: JitMode = "script",
    inputs_path: str | None = None,
    save_path: str | None = None,
    device: DeviceLike = torch.device("cpu"),
    is_half: bool = False,
) -> dict:
    if not save_path:
        save_path = model_path.rstrip(".pth")
        save_path += ".half.jit" if is_half else ".jit"
    if "cuda" in str(device) and ":" not in str(device):
        device = torch.device("cuda:0")
    from .get_synthesizer import get_synthesizer

    model, cpt = get_synthesizer(model_path, device)
    assert isinstance(cpt, dict)
    model.forward = model.infer
    inputs: TensorInputs | None = None
    if mode == "trace":
        assert inputs_path is not None
        inputs = load_inputs(inputs_path, device, is_half)
    ckpt = export(model, mode, inputs, device, is_half)
    cpt.pop("weight")
    cpt["model"] = ckpt["model"]
    cpt["device"] = device
    save(cpt, save_path)
    return cpt
