import torch
from infer.lib.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM

if __name__ == "__main__":
    MoeVS = True  # Whether the model is used by MoeVoiceStudio (formerly MoeSS)

    ModelPath = "Shiroha/shiroha.pth"  # Model path
    ExportedPath = "model.onnx"  # Output path
    hidden_channels = 256  # hidden_channels, preparing for 768Vec
    cpt = torch.load(ModelPath, map_location="cpu")
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    print(*cpt["config"])

    test_phone = torch.rand(1, 200, hidden_channels)  # hidden unit
    test_phone_lengths = torch.tensor([200]).long()  # Hidden unit length (seems useless)
    test_pitch = torch.randint(size=(1, 200), low=5, high=255)  # Fundamental frequency (in Hertz)
    test_pitchf = torch.rand(1, 200)  # NSF fundamental frequency
    test_ds = torch.LongTensor([0])  # Speaker ID
    test_rnd = torch.rand(1, 192, 200)  # Noise (adding a random factor)

    device = "cpu"  # Export device (doesn't affect model usage)

    net_g = SynthesizerTrnMsNSFsidM(
        *cpt["config"], is_half=False
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
        opset_version=16,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
    )
