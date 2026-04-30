from pathlib import Path
import torch
import onnxsim
import onnx
from infer.lib.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM


def export_onnx(ModelPath: Path, ExportedPath: Path):
    cpt = torch.load(ModelPath, map_location="cpu", weights_only=False)
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    vec_channels = 256 if cpt.get("version", "v1") == "v1" else 768

    test_phone = torch.rand(1, 200, vec_channels)  # hidden unit
    test_phone_lengths = torch.tensor([200]).long()  # Hidden unit length (seems useless)
    test_pitch = torch.randint(size=(1, 200), low=5, high=255)  # Fundamental frequency (in Hertz)
    test_pitchf = torch.rand(1, 200)  # NSF fundamental frequency
    test_ds = torch.LongTensor([0])  # Speaker ID
    test_rnd = torch.rand(1, 192, 200)  # Noise (adding a random factor)

    device = "cpu"  # Export device (doesn't affect model usage)

    net_g = SynthesizerTrnMsNSFsidM(
        *cpt["config"], is_half=False, version=cpt.get("version", "v1")
    )  # Export as fp32 (C++ support for fp16 requires manual memory rearrangement, so we're not using fp16 for now)
    net_g.load_state_dict(cpt["weight"], strict=False)
    input_names = ["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"]
    output_names = [
        "audio",
    ]
    # net_g.construct_spkmixmap(n_speaker) Export mixed tracks for multiple characters
    torch.onnx.export(
        net_g,
        (
            test_phone.to(device),
            test_phone_lengths.to(device),
            test_pitch.to(device),
            test_pitchf.to(device),
            test_ds.to(device),
            test_rnd.to(device),
        ),
        ExportedPath,
        dynamic_axes={
            "phone": [1],
            "pitch": [1],
            "pitchf": [1],
            "rnd": [2],
        },
        do_constant_folding=False,
        opset_version=18,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
    )
    model, _ = onnxsim.simplify(str(ExportedPath))
    onnx.save(model, str(ExportedPath))
    return "Finished"
