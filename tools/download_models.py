import os
from pathlib import Path
import requests

RVC_DOWNLOAD_LINK = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/"

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
PRETRAINED_DIR = ASSETS_DIR / "pretrained"
PRETRAINED_V2_DIR = ASSETS_DIR / "pretrained_v2"
UVR5_WEIGHTS_DIR = ASSETS_DIR / "uvr5_weights"

MODEL_NAMES = [
    "D32k.pth",
    "D40k.pth",
    "D48k.pth",
    "G32k.pth",
    "G40k.pth",
    "G48k.pth",
    "f0D32k.pth",
    "f0D40k.pth",
    "f0D48k.pth",
    "f0G32k.pth",
    "f0G40k.pth",
    "f0G48k.pth",
]


def download_model(link, model_name, dir_name):
    with requests.get(f"{link}{model_name}") as r:
        r.raise_for_status()
        os.makedirs(os.path.dirname(dir_name / model_name), exist_ok=True)
        with open(dir_name / model_name, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


if __name__ == "__main__":
    print("Downloading models")

    # Download hubert_base.pt, rmvpe.pt, rmvpe.onnx
    for model in ["hubert_base.pt", "rmvpe.pt", "rmvpe.onnx"]:
        print(f"Downloading {model}...")
        download_model(RVC_DOWNLOAD_LINK, model, ASSETS_DIR / model.split(".")[0])

    # Download pretrained models
    for dir_name in [PRETRAINED_DIR, PRETRAINED_V2_DIR]:
        print(f"\nDownloading {dir_name.name} models:")
        for model in MODEL_NAMES:
            print(f"Downloading {model}...")
            download_model(RVC_DOWNLOAD_LINK + f"{dir_name.name}/", model, dir_name)

    # Download vocals.onnx
    print("Downloading vocals.onnx")
    download_model(
        RVC_DOWNLOAD_LINK + "uvr5_weights/onnx_dereverb_By_FoxJoy/",
        "vocals.onnx",
        ASSETS_DIR / "uvr5_weights/onnx_dereverb_By_FoxJoy",
    )

    # Download uvr5_weights
    print("\nDownloading uvr5_weights")
    for model in [
        "HP2-%E4%BA%BA%E5%A3%B0vocals%2B%E9%9D%9E%E4%BA%BA%E5%A3%B0instrumentals.pth",
        "HP2_all_vocals.pth",
        "HP3_all_vocals.pth",
        "HP5-%E4%B8%BB%E6%97%8B%E5%BE%8B%E4%BA%BA%E5%A3%B0vocals%2B%E5%85%B6%E4%BB%96instrumentals.pth",
        "HP5_only_main_vocal.pth",
        "VR-DeEchoAggressive.pth",
        "VR-DeEchoDeReverb.pth",
        "VR-DeEchoNormal.pth",
    ]:
        print(f"Downloading {model}...")
        download_model(RVC_DOWNLOAD_LINK + "uvr5_weights/", model, UVR5_WEIGHTS_DIR)

    print("\nAll models downloaded!")
