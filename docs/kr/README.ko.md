<div align="center">

<h1>Retrieval-based-Voice-Conversion-WebUI</h1>
간단하고 사용하기 쉬운 음색 변환/보이스 체인저 프레임워크.<br><br>

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

<img src="https://counter.seku.su/cmoe?name=rvc&theme=r34" /><br>

[![Licence](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-Models-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)


[**업데이트 로그**](./Changelog_KO.md) | [**자주 묻는 질문**](./faq_ko.md) | [**AutoDL·5원으로 AI 가수 훈련**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/Autodl%E8%AE%AD%E7%BB%83RVC%C2%B7AI%E6%AD%8C%E6%89%8B%E6%95%99%E7%A8%8B) | [**대조 실험 기록**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/%E5%AF%B9%E7%85%A7%E5%AE%9E%E9%AA%8C%C2%B7%E5%AE%9E%E9%AA%8C%E8%AE%B0%E5%BD%95) | [**온라인 데모**](https://modelscope.cn/studios/FlowerCry/RVCv2demo)

[**English**](../en/README.en.md) | [**中文简体**](../../README.md) | [**日本語**](../jp/README.ja.md) | [**한국어**](../kr/README.ko.md) ([**韓國語**](../kr/README.ko.han.md)) | [**Français**](../fr/README.fr.md) | [**Türkçe**](../tr/README.tr.md) | [**Português**](../pt/README.pt.md)

</div>

> [데모 영상](https://www.bilibili.com/video/BV1pm4y1z7Gm/)을 확인해 보세요!

> RVC를 활용한 실시간 음성변환: [w-okada/voice-changer](https://github.com/w-okada/voice-changer)

> 기본 모델은 50시간 가량의 고퀄리티 오픈 소스 VCTK 데이터셋을 사용하였으므로, 저작권상의 염려가 없으니 안심하고 사용하시기 바랍니다.

> 더 큰 매개변수, 더 큰 데이터, 더 나은 효과, 기본적으로 동일한 추론 속도, 더 적은 양의 훈련 데이터가 필요한 RVCv3의 기본 모델을 기대해 주십시오.

<table>
   <tr>
		<td align="center">훈련 및 추론 인터페이스</td>
		<td align="center">실시간 음성 변환 인터페이스</td>
	</tr>
  <tr>
		<td align="center"><img src="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/assets/129054828/092e5c12-0d49-4168-a590-0b0ef6a4f630"></td>
    <td align="center"><img src="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/assets/129054828/730b4114-8805-44a1-ab1a-04668f3c30a6"></td>
	</tr>
	<tr>
		<td align="center">go-webui.bat</td>
		<td align="center">go-realtime_gui.bat</td>
	</tr>
  <tr>
    <td align="center">원하는 작업을 자유롭게 선택할 수 있습니다.</td>
		<td align="center">우리는 이미 끝에서 끝까지 170ms의 지연을 실현했습니다. ASIO 입력 및 출력 장치를 사용하면 끝에서 끝까지 90ms의 지연을 달성할 수 있지만, 이는 하드웨어 드라이버 지원에 매우 의존적입니다.</td>
	</tr>
</table>

## 소개

본 Repo는 다음과 같은 특징을 가지고 있습니다:

- top1 검색을 이용하여 입력 음색 특징을 훈련 세트 음색 특징으로 대체하여 음색의 누출을 방지
- 상대적으로 낮은 성능의 GPU에서도 빠른 훈련 가능
- 적은 양의 데이터로 훈련해도 좋은 결과를 얻을 수 있음 (최소 10분 이상의 저잡음 음성 데이터를 사용하는 것을 권장)
- 모델 융합을 통한 음색의 변조 가능 (ckpt 처리 탭->ckpt 병합 선택)
- 사용하기 쉬운 WebUI (웹 인터페이스)
- UVR5 모델을 이용하여 목소리와 배경음악의 빠른 분리;
- 최첨단 [음성 피치 추출 알고리즘 InterSpeech2023-RMVPE](#参考项目)을 사용하여 무성음 문제를 해결합니다. 효과는 최고(압도적)이며 crepe_full보다 더 빠르고 리소스 사용이 적음
- A카드와 I카드 가속을 지원

해당 프로젝트의 [데모 비디오](https://www.bilibili.com/video/BV1pm4y1z7Gm/)를 확인해보세요!

## 환경 설정

이 브랜치는 **Python 3.12 x64**를 대상으로 합니다. 모든 명령은 저장소 루트에서 실행하세요. Ubuntu 24.04 x86_64를 권장합니다.

### Ubuntu 24.04

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev ffmpeg unzip libsndfile1 libportaudio2

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### Windows

Python 3.12 x64를 설치한 다음 가상 환경을 만드세요.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
```

### 하드웨어별 의존성 선택

| 하드웨어 | 설치 방법 |
| --- | --- |
| CPU, AMD, Intel | `requirments_cpu_py312.txt` 사용. Windows는 DirectML, Linux는 CPU 사용 |
| NVIDIA RTX 50 시리즈 | CUDA 12.8 Torch를 먼저 설치한 뒤 `requirments_cu128_py312.txt` 설치 |
| RTX 50 시리즈 이전 NVIDIA | CUDA 11.8 Torch를 먼저 설치한 뒤 `requirments_cu118_py312.txt` 설치 |

#### CPU, AMD, Intel

```bash
python -m pip install -r requirments_cpu_py312.txt
```

#### NVIDIA RTX 50 시리즈: 2단계 설치

```bash
python -m pip install torch==2.7.1+cu128 torchaudio==2.7.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple
python -m pip install -r requirments_cu128_py312.txt
```

#### RTX 50 시리즈 이전 NVIDIA: 2단계 설치

```bash
python -m pip install torch==2.7.1+cu118 torchaudio==2.7.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118 \
  --extra-index-url https://pypi.org/simple
python -m pip install -r requirments_cu118_py312.txt
```

Torch와 CUDA 상태를 확인하세요.

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available())"
```

프로그램은 NVIDIA GPU 메모리와 연산 능력도 확인합니다. 약 4 GiB 미만이거나 SM 5.3 미만이면 CPU 경로를 사용합니다.

### 패키지 다운로드 소스

세 개의 `requirments_*.txt` 파일 맨 위에 다운로드 소스가 있습니다. 공식 소스를 사용할 때는 `--index-url`과 `--extra-index-url`만 교체하고 버전, CUDA 접미사, 2단계 순서는 유지하세요.

| Default mirror | Official source |
| --- | --- |
| `https://mirrors.pku.edu.cn/pypi/simple` | `https://pypi.org/simple` |
| `https://mirrors.nju.edu.cn/pytorch/whl/cpu` | `https://download.pytorch.org/whl/cpu` |
| `https://mirrors.nju.edu.cn/pytorch/whl/cu118` | `https://download.pytorch.org/whl/cu118` |
| `https://mirrors.nju.edu.cn/pytorch/whl/cu128` | `https://download.pytorch.org/whl/cu128` |

## 모델 및 실행 디렉터리

WebUI는 실행 디렉터리를 자동 생성합니다. [Hugging Face 모델 저장소](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main)에서 모델을 받고 다음 구조를 유지하세요.

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

### 모델 다운로드

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

이전 형식인 `hubert_base.pt`는 이 브랜치에서 사용하지 않습니다. 현재 코드는 `assets/hubert_base/`의 Transformers 모델을 사용합니다. FCPE는 `torchfcpe`에 포함됩니다.

### FFmpeg

위 Ubuntu 명령은 FFmpeg를 설치합니다. Windows에서는 다음 파일을 저장소 루트에 배치하세요.

- [ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/ffmpeg.exe?download=true)
- [ffprobe.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/ffprobe.exe?download=true)

## WebUI 시작

```bash
python webui.py
```

화면이 없는 Ubuntu 서버:

```bash
python webui.py --noautoopen
```

기본 포트는 `7865`입니다. `.pth` 모델은 `assets/weights/`, `.index` 파일은 `assets/indices/`에 배치하세요.

## 참조 프로젝트

- [ContentVec](https://github.com/auspicious3000/contentvec/)
- [VITS](https://github.com/jaywalnut310/vits)
- [HIFIGAN](https://github.com/jik876/hifi-gan)
- [Gradio](https://github.com/gradio-app/gradio)
- [FFmpeg](https://github.com/FFmpeg/FFmpeg)
- [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
- [audio-slicer](https://github.com/openvpi/audio-slicer)
- [Vocal pitch extraction:RMVPE](https://github.com/Dream-High/RMVPE)
  - 사전 훈련된 모델은 [yxlllc](https://github.com/yxlllc/RMVPE)와 [RVC-Boss](https://github.com/RVC-Boss)에 의해 훈련되고 테스트되었습니다.

## 모든 기여자들의 노력에 감사드립니다

<a href="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=RVC-Project/Retrieval-based-Voice-Conversion-WebUI" />
</a>
