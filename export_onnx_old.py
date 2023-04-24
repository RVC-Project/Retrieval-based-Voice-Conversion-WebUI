from infer_pack.models_onnx_moess import SynthesizerTrnMs256NSFsidM
import torch

person = "Shiroha/shiroha.pth"
exported_path = "model.onnx"


cpt = torch.load(person, map_location="cpu")
cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
print(*cpt["config"])
net_g = SynthesizerTrnMs256NSFsidM(*cpt["config"], is_half=False)
net_g.load_state_dict(cpt["weight"], strict=False)

test_phone = torch.rand(1, 200, 256)
test_phone_lengths = torch.tensor([200]).long()
test_pitch = torch.randint(size=(1, 200), low=5, high=255)
test_pitchf = torch.rand(1, 200)
test_ds = torch.LongTensor([0])
test_rnd = torch.rand(1, 192, 200)
input_names = ["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"]
output_names = [
    "audio",
]
device = "cpu"
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
    exported_path,
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
