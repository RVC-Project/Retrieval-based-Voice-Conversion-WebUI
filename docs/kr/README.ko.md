<div align="center">

<h1>Retrieval-based-Voice-Conversion-WebUI</h1>
VITS 기반의 간단하고 사용하기 쉬운 음성 변환 프레임워크.<br><br>

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange
)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

<img src="https://counter.seku.su/cmoe?name=rvc&theme=r34" /><br>

[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/Retrieval_based_Voice_Conversion_WebUI.ipynb)
[![Licence](https://img.shields.io/github/license/RVC-Project/Retrieval-based-Voice-Conversion-WebUI?style=for-the-badge)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-Spaces-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)

[![Discord](https://img.shields.io/badge/RVC%20Developers-Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/HcsmBBGyVk)

</div>

---

[**업데이트 로그**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/Changelog_KO.md)

[**English**](../en/README.en.md) | [**中文简体**](../../README.md) | [**日本語**](../jp/README.ja.md) | [**한국어**](../kr/README.ko.md) ([**韓國語**](../kr/README.ko.han.md)) | [**Türkçe**](../tr/README.tr.md)

> [데모 영상](https://www.bilibili.com/video/BV1pm4y1z7Gm/)을 확인해 보세요!

> RVC를 활용한 실시간 음성변환: [w-okada/voice-changer](https://github.com/w-okada/voice-changer)

> 기본 모델은 50시간 가량의 고퀄리티 오픈 소스 VCTK 데이터셋을 사용하였으므로, 저작권상의 염려가 없으니 안심하고 사용하시기 바랍니다.

> 저작권 문제가 없는 고퀄리티의 노래를 이후에도 계속해서 훈련할 예정입니다.

## 소개

본 Repo는 다음과 같은 특징을 가지고 있습니다:

- top1 검색을 이용하여 입력 음색 특징을 훈련 세트 음색 특징으로 대체하여 음색의 누출을 방지;
- 상대적으로 낮은 성능의 GPU에서도 빠른 훈련 가능;
- 적은 양의 데이터로 훈련해도 좋은 결과를 얻을 수 있음 (최소 10분 이상의 저잡음 음성 데이터를 사용하는 것을 권장);
- 모델 융합을 통한 음색의 변조 가능 (ckpt 처리 탭->ckpt 병합 선택);
- 사용하기 쉬운 WebUI (웹 인터페이스);
- UVR5 모델을 이용하여 목소리와 배경음악의 빠른 분리;

## 환경의 준비

poetry를 통해 dependecies를 설치하는 것을 권장합니다.

다음 명령은 Python 버전 3.8 이상의 환경에서 실행되어야 합니다:

```bash
# PyTorch 관련 주요 dependencies 설치, 이미 설치되어 있는 경우 건너뛰기 가능
# 참조: https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio

# Windows + Nvidia Ampere Architecture(RTX30xx)를 사용하고 있다면, https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/issues/21 에서 명시된 것과 같이 PyTorch에 맞는 CUDA 버전을 지정해야 합니다.
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Poetry 설치, 이미 설치되어 있는 경우 건너뛰기 가능
# Reference: https://python-poetry.org/docs/#installation
curl -sSL https://install.python-poetry.org | python3 -

# Dependecies 설치
poetry install
```

pip를 활용하여 dependencies를 설치하여도 무방합니다.

```bash
pip install -r requirements.txt
```

## 기타 사전 모델 준비

RVC 모델은 추론과 훈련을 위하여 다른 사전 모델이 필요합니다.

[Huggingface space](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)를 통해서 다운로드 할 수 있습니다.

다음은 RVC에 필요한 사전 모델 및 기타 파일 목록입니다:

```bash
./assets/hubert/hubert_base.pt

./assets/pretrained 

./assets/uvr5_weights

V2 버전 모델을 테스트하려면 추가 다운로드가 필요합니다.

./assets/pretrained_v2

# Windows를 사용하는 경우 이 사전도 필요할 수 있습니다. FFmpeg가 설치되어 있으면 건너뛰어도 됩니다.
ffmpeg.exe
```

그 후 이하의 명령을 사용하여 WebUI를 시작할 수 있습니다:

```bash
python infer-web.py
```

Windows를 사용하는 경우 `RVC-beta.7z`를 다운로드 및 압축 해제하여 RVC를 직접 사용하거나 `go-web.bat`을 사용하여 WebUi를 시작할 수 있습니다.

## 참고

- [ContentVec](https://github.com/auspicious3000/contentvec/)
- [VITS](https://github.com/jaywalnut310/vits)
- [HIFIGAN](https://github.com/jik876/hifi-gan)
- [Gradio](https://github.com/gradio-app/gradio)
- [FFmpeg](https://github.com/FFmpeg/FFmpeg)
- [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
- [audio-slicer](https://github.com/openvpi/audio-slicer)

## 모든 기여자 분들의 노력에 감사드립니다.

<a href="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=RVC-Project/Retrieval-based-Voice-Conversion-WebUI" />
</a>
