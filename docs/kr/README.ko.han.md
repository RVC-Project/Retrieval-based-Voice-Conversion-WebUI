<div align="center">

<h1>Retrieval-based-Voice-Conversion-WebUI</h1>
VITS基盤의 簡單하고使用하기 쉬운音聲變換틀<br><br>

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange
)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

<img src="https://counter.seku.su/cmoe?name=rvc&theme=r34" /><br>
  
[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/Retrieval_based_Voice_Conversion_WebUI.ipynb)
[![Licence](https://img.shields.io/github/license/RVC-Project/Retrieval-based-Voice-Conversion-WebUI?style=for-the-badge)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-Spaces-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)

[![Discord](https://img.shields.io/badge/RVC%20Developers-Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/HcsmBBGyVk)

</div>

------
[**更新日誌**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/Changelog_KO.md)

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

## 環境의準備
poetry를通해 依存를設置하는 것을 勸獎합니다.

다음命令은 Python 버전3.8以上의環境에서 實行되어야 합니다:
```bash
# PyTorch 關聯主要依存設置, 이미設置되어 있는 境遇 건너뛰기 可能
# 參照: https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio

# Windows + Nvidia Ampere Architecture(RTX30xx)를 使用하고 있다面, #21 에서 명시된 것과 같이 PyTorch에 맞는 CUDA 버전을 指定해야 합니다.
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Poetry 設置, 이미設置되어 있는 境遇 건너뛰기 可能
# Reference: https://python-poetry.org/docs/#installation
curl -sSL https://install.python-poetry.org | python3 -

# 依存設置
poetry install
```
pip를 活用하여依存를 設置하여도 無妨합니다.

```bash
pip install -r requirements.txt
```

## 其他預備모델準備
RVC 모델은 推論과訓練을 依하여 다른 預備모델이 必要합니다.

[Huggingface space](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)를 通해서 다운로드 할 수 있습니다.

다음은 RVC에 必要한 預備모델 및 其他 파일 目錄입니다:
```bash
./assets/hubert_base

./assets/pretrained 

./assets/uvr5_weights

V2 버전 모델을 테스트하려면 추가 다운로드가 필요합니다.

./assets/pretrained_v2

# Windows를 使用하는境遇 이 사전도 必要할 수 있습니다. FFmpeg가 設置되어 있으면 건너뛰어도 됩니다.
ffmpeg.exe
```
그後 以下의 命令을 使用하여 WebUI를 始作할 수 있습니다:
```bash
python webui.py
```
Windows를 使用하는境遇 `RVC-beta.7z`를 다운로드 및 壓縮解除하여 RVC를 直接使用하거나 `go-webui.bat`을 使用하여 WebUi를 直接할 수 있습니다.

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
