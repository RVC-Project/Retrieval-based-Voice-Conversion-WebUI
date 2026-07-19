<div align="center">

<h1>Retrieval-based-Voice-Conversion-WebUI</h1>
簡單하고 使用하기 쉬운 音色變換/變聲器 프레임워크.<br><br>

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange
)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

<img src="https://counter.seku.su/cmoe?name=rvc&theme=r34" /><br>

[![Licence](https://img.shields.io/github/license/RVC-Project/Retrieval-based-Voice-Conversion-WebUI?style=for-the-badge)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-Models-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)


</div>

------
[**更新日誌**](./Changelog_KO.md)

[**English**](../en/README.en.md) | [**中文简体**](../../README.md) | [**日本語**](../jp/README.ja.md) | [**한국어**](../kr/README.ko.md) ([**韓國語**](../kr/README.ko.han.md)) | [**Français**](../fr/README.fr.md) | [**Türkçe**](../tr/README.tr.md) | [**Português**](../pt/README.pt.md)

> [示範映像](https://www.bilibili.com/video/BV1pm4y1z7Gm/)을 確認해 보세요!

> RVC를活用한實時間音聲變換: [w-okada/voice-changer](https://github.com/w-okada/voice-changer)

> 基本모델은 50時間假量의 高品質 오픈 소스 VCTK 데이터셋을 使用하였으므로, 著作權上의 念慮가 없으니 安心하고 使用하시기 바랍니다.

> 著作權問題가 없는 高品質의 노래를 以後에도 繼續해서 訓練할 豫定입니다.

## 紹介
本Repo는 다음과 같은 特徵을 가지고 있습니다:
+ top1檢索을利用하여 入力音色特徵을 訓練세트音色特徵으로 代替하여 音色의漏出을 防止;
+ 相對的으로 낮은性能의 GPU에서도 빠른訓練可能;
+ 적은量의 데이터로 訓練해도 좋은 結果를 얻을 수 있음 (最小10分以上의 低雜음音聲데이터를 使用하는 것을 勸獎);
+ 모델融合을通한 音色의 變調可能 (ckpt處理탭->ckpt混合選擇);
+ 使用하기 쉬운 WebUI (웹 使用者인터페이스);
+ UVR5 모델을 利用하여 목소리와 背景音樂의 빠른 分離;

## 環境의 準備

이 브랜치는 **Python 3.12 x64**를 對象으로 합니다. 모든 命令은 저장소 根目錄에서 실행하세요. Ubuntu 24.04 x86_64를 권장합니다.

### Ubuntu 24.04

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev ffmpeg unzip libsndfile1 libportaudio2

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### Windows

Python 3.12 x64를 설치한 뒤 假想環境을 만드세요.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
```

### 하드웨어別 依存性

| 하드웨어 | 設置 方法 |
| --- | --- |
| CPU, AMD, Intel | `requirments_cpu_py312.txt` 使用. Windows는 DirectML, Linux는 CPU 使用 |
| NVIDIA RTX 50 系列 | CUDA 12.8 Torch를 먼저 설치한 뒤 `requirments_cu128_py312.txt` 設置 |
| RTX 50 系列 以前 NVIDIA | CUDA 11.8 Torch를 먼저 설치한 뒤 `requirments_cu118_py312.txt` 設置 |

#### CPU, AMD, Intel

```bash
python -m pip install -r requirments_cpu_py312.txt
```

#### NVIDIA RTX 50 系列：2段階 設置

```bash
python -m pip install torch==2.7.1+cu128 torchaudio==2.7.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple
python -m pip install -r requirments_cu128_py312.txt
```

#### RTX 50 系列 以前 NVIDIA：2段階 設置

Torch와 CUDA 狀態를 確認하세요.

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available())"
```

프로그램은 NVIDIA GPU 記憶體와 計算能力도 확인합니다. 약 4 GiB 미만이거나 SM 5.3 미만이면 CPU 經路를 사용합니다.

### 패키지 來源

세 개의 `requirments_*.txt` 上端에 下載 來源이 있습니다. 공식 소스를 사용할 때는 `--index-url`과 `--extra-index-url`만 교체하고 버전, CUDA 접미사, 2段階 순서는 유지하세요.

| Default mirror | Official source |
| --- | --- |
| `https://mirrors.pku.edu.cn/pypi/simple` | `https://pypi.org/simple` |
| `https://mirrors.nju.edu.cn/pytorch/whl/cpu` | `https://download.pytorch.org/whl/cpu` |
| `https://mirrors.nju.edu.cn/pytorch/whl/cu118` | `https://download.pytorch.org/whl/cu118` |
| `https://mirrors.nju.edu.cn/pytorch/whl/cu128` | `https://download.pytorch.org/whl/cu128` |

## 모델과 運行 目錄

WebUI는 運行 目錄를 자동 생성합니다. [Hugging Face 모델 저장소](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main)에서 모델을 받고 다음 구조를 유지하세요.

```text
assets/
├── hubert_base/
│   ├── config.json
│   ├── preprocessor_config.json
│   └── pytorch_model.bin
├── rmvpe/rmvpe.pt
├── pretrained/
├── pretrained_v2/
├── uvr5_weights/
├── weights/        # user RVC .pth models
└── indices/        # user .index files
logs/
└── mute/           # training silence samples

# Exact paths used by the code
assets/hubert_base/config.json
assets/hubert_base/preprocessor_config.json
assets/hubert_base/pytorch_model.bin
assets/rmvpe/rmvpe.pt
assets/pretrained/*.pth
assets/pretrained_v2/*.pth
assets/uvr5_weights/*
assets/weights/*.pth
assets/indices/*.index
logs/mute/*
```

### 모델 下載

```bash
python -m pip install --upgrade huggingface_hub

# Required for inference and feature extraction
hf download lj1995/VoiceConversionWebUI --revision main \
  --include "hubert_base/*" --local-dir assets
hf download lj1995/VoiceConversionWebUI rmvpe.pt --revision main \
  --local-dir assets/rmvpe

# Required for v1/v2 training
hf download lj1995/VoiceConversionWebUI --revision main \
  --include "pretrained/*" "pretrained_v2/*" --local-dir assets
hf download lj1995/VoiceConversionWebUI mute.zip --revision main \
  --local-dir .model-downloads
python -m zipfile -e .model-downloads/mute.zip logs

# Required only for UVR5 vocal separation
hf download lj1995/VoiceConversionWebUI --revision main \
  --include "uvr5_weights/*" --local-dir assets
```

Windows AMD/Intel DirectML 환경에는 다음 파일도 필요합니다.

```bash
hf download lj1995/VoiceConversionWebUI rmvpe.onnx --revision main \
  --local-dir assets/rmvpe
```

舊形式 `hubert_base.pt`는 이 브랜치에서 사용하지 않습니다. 현재 코드는 `assets/hubert_base/`의 Transformers 모델을 사용합니다. FCPE는 `torchfcpe`에 포함됩니다.

### FFmpeg

위 Ubuntu 명령은 FFmpeg를 설치합니다. Windows에서는 다음 파일을 저장소 根目錄에 배치하세요.

- [ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/ffmpeg.exe?download=true)
- [ffprobe.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/ffprobe.exe?download=true)

## WebUI 始作

```bash
python webui.py
```

화면이 없는 Ubuntu 서버:

```bash
python webui.py --noautoopen
```

基本 포트는 `7865`입니다. `.pth` 모델은 `assets/weights/`, `.index` 파일은 `assets/indices/`에 배치하세요.

## 參考
+ [ContentVec](https://github.com/auspicious3000/contentvec/)
+ [VITS](https://github.com/jaywalnut310/vits)
+ [HIFIGAN](https://github.com/jik876/hifi-gan)
+ [Gradio](https://github.com/gradio-app/gradio)
+ [FFmpeg](https://github.com/FFmpeg/FFmpeg)
+ [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
+ [audio-slicer](https://github.com/openvpi/audio-slicer)
## 모든寄與者분들의勞力에感謝드립니다

<a href="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=RVC-Project/Retrieval-based-Voice-Conversion-WebUI" />
</a>
