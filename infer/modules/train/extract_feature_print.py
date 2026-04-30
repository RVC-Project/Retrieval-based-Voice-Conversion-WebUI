import os
import sys
import traceback
from pathlib import Path

from tap import Tap

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


class ExtractFeatureCpuArgs(Tap):
    device: str
    n_part: int
    i_part: int
    exp_dir: Path
    version: str
    is_half: bool

    def configure(self) -> None:
        self.add_argument("device")
        self.add_argument("n_part")
        self.add_argument("i_part")
        self.add_argument("exp_dir")
        self.add_argument("version")
        self.add_argument("is_half")


class ExtractFeatureGpuArgs(Tap):
    device: str
    n_part: int
    i_part: int
    i_gpu: str
    exp_dir: Path
    version: str
    is_half: bool

    def configure(self) -> None:
        self.add_argument("device")
        self.add_argument("n_part")
        self.add_argument("i_part")
        self.add_argument("i_gpu")
        self.add_argument("exp_dir")
        self.add_argument("version")
        self.add_argument("is_half")


if len(sys.argv) == 7:
    parsed_args = ExtractFeatureCpuArgs().parse_args()
elif len(sys.argv) == 8:
    parsed_args = ExtractFeatureGpuArgs().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(parsed_args.i_gpu)
else:
    raise ValueError(
        "Expected positional arguments: device n_part i_part [i_gpu] exp_dir version is_half"
    )
n_part = parsed_args.n_part
i_part = parsed_args.i_part
exp_dir = parsed_args.exp_dir
version = parsed_args.version
is_half = parsed_args.is_half
import fairseq
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

f = open(exp_dir / "extract_f0_feature.log", "a+")


def printt(strr):
    print(strr)
    f.write(f"{strr}\n")
    f.flush()


printt(" ".join(sys.argv))
model_path = "assets/hubert/hubert_base.pt"

printt(f"exp_dir: {exp_dir}")
wavPath = exp_dir / "1_16k_wavs"
outPath = exp_dir / "3_feature256" if version == "v1" else exp_dir / "3_feature768"
outPath.mkdir(parents=True, exist_ok=True)


# wave must be 16k, hop_size=320
def readwave(wav_path, normalize=False):
    wav, sr = sf.read(wav_path)
    assert sr == 16000
    feats = torch.from_numpy(wav).float()
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    feats = feats.view(1, -1)
    return feats


# HuBERT model
printt(f"load model(s) from {model_path}")
# if hubert model is exist
if os.access(model_path, os.F_OK) == False:
    printt(
        f"Error: Extracting is shut down because {model_path} does not exist, you may download it from https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main"
    )
    exit(0)

from fairseq.data.dictionary import Dictionary
from torch.serialization import safe_globals

with safe_globals([Dictionary]):
    # torch.serialization.add_safe_globals([Dictionary])
    models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [model_path],
        suffix="",
    )
model = models[0]
model = model.to(device)
printt(f"move model to {device}")
if is_half:
    if device not in ["mps", "cpu"]:
        model = model.half()
model.eval()

todo = sorted(wavPath.iterdir(), key=lambda p: p.name)[i_part::n_part]
n = max(1, len(todo) // 10)  # Print up to ten entries
if len(todo) == 0:
    printt("no-feature-todo")
else:
    printt(f"all-feature-{len(todo)}")
    if saved_cfg is None:
        raise RuntimeError("HuBERT checkpoint did not include a saved config")
    normalize = saved_cfg.task.normalize
    for idx, file in enumerate(todo):
        try:
            if file.suffix == ".wav":
                wav_path = wavPath / file.name
                out_path = outPath / file.with_suffix(".npy").name

                if out_path.exists():
                    continue

                feats = readwave(wav_path, normalize=normalize)
                padding_mask = torch.BoolTensor(feats.shape).fill_(False)
                inputs = {
                    "source": (
                        feats.half().to(device)
                        if is_half and device not in ["mps", "cpu"]
                        else feats.to(device)
                    ),
                    "padding_mask": padding_mask.to(device),
                    "output_layer": 9 if version == "v1" else 12,  # layer 9
                }
                with torch.no_grad():
                    logits = model.extract_features(**inputs)
                    feats = (
                        model.final_proj(logits[0]) if version == "v1" else logits[0]
                    )

                feats = feats.squeeze(0).float().cpu().numpy()
                if np.isnan(feats).sum() == 0:
                    np.save(out_path, feats, allow_pickle=False)
                else:
                    printt(f"{file.name}-contains nan")
                if idx % n == 0:
                    printt(f"now-{len(todo)},all-{idx},{file.name},{feats.shape}")
        except:
            printt(traceback.format_exc())
    printt("all-feature-done")
