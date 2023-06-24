<div align="center">

<h1>Retrieval-based-Voice-Conversion-WebUI</h1>
VITSに基づく使いやすい音声変換（voice changer）framework<br><br>

[![madewithlove](https://forthebadge.com/images/badges/built-with-love.svg)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

<img src="https://counter.seku.su/cmoe?name=rvc&theme=r34" /><br>

[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/Retrieval_based_Voice_Conversion_WebUI.ipynb)
[![Licence](https://img.shields.io/github/license/RVC-Project/Retrieval-based-Voice-Conversion-WebUI?style=for-the-badge)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-Spaces-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)

[![Discord](https://img.shields.io/badge/RVC%20Developers-Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/HcsmBBGyVk)

</div>

------

[**更新日誌**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/Changelog_CN.md)

[**English**](./README.en.md) | [**中文简体**](../README.md) | [**日本語**](./README.ja.md) | [**한국어**](./README.ko.md) ([**韓國語**](./README.ko.han.md))

> デモ動画は[こちら](https://www.bilibili.com/video/BV1pm4y1z7Gm/)でご覧ください。

> RVCによるリアルタイム音声変換: [w-okada/voice-changer](https://github.com/w-okada/voice-changer)

> 著作権侵害を心配することなく使用できるように、基底モデルは約50時間の高品質なオープンソースデータセットで訓練されています。

> 今後も、次々と使用許可のある高品質な歌声の資料集を追加し、基底モデルを訓練する予定です。

## はじめに
本リポジトリには下記の特徴があります。

+ Top1検索を用いることで、生の特徴量を訓練用データセット特徴量に変換し、トーンリーケージを削減します。
+ 比較的貧弱なGPUでも、高速かつ簡単に訓練できます。
+ 少量のデータセットからでも、比較的良い結果を得ることができます。（10分以上のノイズの少ない音声を推奨します。）
+ モデルを融合することで、音声を混ぜることができます。（ckpt processingタブの、ckpt mergeを使用します。）
+ 使いやすいWebUI。
+ UVR5 Modelも含んでいるため、人の声とBGMを素早く分離できます。

## 環境構築
Poetryで依存関係をインストールすることをお勧めします。

下記のコマンドは、Python3.8以上の環境で実行する必要があります:
```bash
# PyTorch関連の依存関係をインストール。インストール済の場合は省略。
# 参照先: https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio

#Windows＋ Nvidia Ampere Architecture(RTX30xx)の場合、 #21 に従い、pytorchに対応するcuda versionを指定する必要があります。
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# PyTorch関連の依存関係をインストール。インストール済の場合は省略。
# 参照先: https://python-poetry.org/docs/#installation
curl -sSL https://install.python-poetry.org | python3 -

# Poetry経由で依存関係をインストール
poetry install
```

pipでも依存関係のインストールが可能です:

```bash
pip install -r requirements.txt
```

## 基底modelsを準備
RVCは推論/訓練のために、様々な事前訓練を行った基底モデルを必要とします。

modelsは[Hugging Face space](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)からダウンロードできます。

以下は、RVCに必要な基底モデルやその他のファイルの一覧です。
```bash
hubert_base.pt

./pretrained 

./uvr5_weights

# ffmpegがすでにinstallされている場合は省略
./ffmpeg
```
その後、下記のコマンドでWebUIを起動します。
```bash
python infer-web.py
```
Windowsをお使いの方は、直接`RVC-beta.7z`をダウンロード後に展開し、`go-web.bat`をクリックすることで、WebUIを起動することができます。(7zipが必要です。)

また、リポジトリに[小白简易教程.doc](./小白简易教程.doc)がありますので、参考にしてください（中国語版のみ）。

## 参考プロジェクト
+ [ContentVec](https://github.com/auspicious3000/contentvec/)
+ [VITS](https://github.com/jaywalnut310/vits)
+ [HIFIGAN](https://github.com/jik876/hifi-gan)
+ [Gradio](https://github.com/gradio-app/gradio)
+ [FFmpeg](https://github.com/FFmpeg/FFmpeg)
+ [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
+ [audio-slicer](https://github.com/openvpi/audio-slicer)

## 貢献者(contributor)の皆様の尽力に感謝します
<a href="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=RVC-Project/Retrieval-based-Voice-Conversion-WebUI" />
</a>
