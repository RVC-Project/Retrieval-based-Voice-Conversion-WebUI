from pathlib import Path
import torch

from .synthesizer import SynthesizerTrnMsNSFsid


def export_onnx(from_cpkt_pth: Path, to_onnx_pth: Path) -> str:
    cpt = torch.load(from_cpkt_pth, map_location="cpu")
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    vec_channels = 256 if cpt.get("version", "v1") == "v1" else 768

    test_phone = torch.rand(1, 200, vec_channels)  # hidden unit
    test_phone_lengths = torch.tensor([200]).long()  # Hidden unit length (seems useless)
    test_pitch = torch.randint(size=(1, 200), low=5, high=255)  # Fundamental frequency (in Hertz)
    test_pitchf = torch.rand(1, 200)  # NSF fundamental frequency
    test_ds = torch.LongTensor([0])  # Speaker ID
    test_rnd = torch.rand(1, 192, 200)  # Noise (adding a random factor)

    device = "cpu"  # Export device (doesn't affect model usage)

    net_g = SynthesizerTrnMsNSFsid(
        *cpt["config"], encoder_dim=vec_channels
    )  # Export as fp32 (C++ support for fp16 requires manual memory rearrangement, so we're not using fp16 for now)
    net_g.load_state_dict(cpt["weight"], strict=False)
    input_names = ["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"]
    output_names = [
        "audio",
    ]
    # net_g.construct_spkmixmap() #Export mixed tracks for multiple characters
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
        to_onnx_pth,
        dynamic_axes={
            "phone": [1],
            "pitch": [1],
            "pitchf": [1],
            "rnd": [2],
        },
        do_constant_folding=False,
        opset_version=17,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
    )
    return "Finished"
