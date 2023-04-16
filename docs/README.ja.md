<div align="center">

<h1>Retrieval-based-Voice-Conversion-WebUI</h1>
VITSに基づく使いやすい音声変換（voice changer）framework<br><br>

[![madewithlove](https://forthebadge.com/images/badges/built-with-love.svg)](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI)

<img src="https://counter.seku.su/cmoe?name=rvc&theme=r34" /><br>

[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/liujing04/Retrieval-based-Voice-Conversion-WebUI/blob/main/Retrieval_based_Voice_Conversion_WebUI.ipynb)
[![Licence](https://img.shields.io/github/license/liujing04/Retrieval-based-Voice-Conversion-WebUI?style=for-the-badge)](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI/blob/main/%E4%BD%BF%E7%94%A8%E9%9C%80%E9%81%B5%E5%AE%88%E7%9A%84%E5%8D%8F%E8%AE%AE-LICENSE.txt)
[![Huggingface](https://img.shields.io/badge/🤗%20-Spaces-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)

[![Discord](https://img.shields.io/badge/RVC%20Developers-Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/HcsmBBGyVk)

</div>

------

[**更新日誌**](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI/blob/main/Changelog_CN.md)

[**English**](./README.en.md) | [**中文简体**](../README.md) | [**日本語**](./README.ja.md)

> デモ動画は[こちら](https://www.bilibili.com/video/BV1pm4y1z7Gm/)でご覧ください

> RVCによるリアルタイム音声変換: [w-okada/voice-changer](https://github.com/w-okada/voice-changer)

> 基底modelを訓練(training)したのは、約50時間の高品質なオープンソースのデータセット。著作権侵害を心配することなく使用できるように。

> 今後は次々と使用許可のある高品質歌声資料集を追加し、基底modelを訓練する。

## はじめに
本repoは下記の特徴があります

+ 調子(tone)の漏洩が下がれるためtop1検索で源特徴量を訓練集特徴量に置換
+ 古い又は安いGPUでも高速に訓練できる
+ 小さい訓練集でもかなりいいmodelを得られる(10分以上の低noise音声を推奨)
+ modelを融合し音色をmergeできる(ckpt processing->ckpt mergeで使用)
+ 使いやすいWebUI
+ UVR5 Modelも含めるため人声とBGMを素早く分離できる

## 環境構築
poetryで依存関係をinstallすることをお勧めします。

下記のcommandsは、Python3.8以上の環境で実行する必要があります:
```bash
# PyTorch関連の依存関係をinstall。install済の場合はskip
# 参照先: https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio

#Windows＋ Nvidia Ampere Architecture(RTX30xx)の場合、 #21 に従い、pytorchに対応するcuda versionを指定する必要があります。
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# PyTorch関連の依存関係をinstall。install済の場合はskip
# 参照先: https://python-poetry.org/docs/#installation
curl -sSL https://install.python-poetry.org | python3 -

# Poetry経由で依存関係をinstall
poetry install
```

pipでも依存関係のinstallが可能です:

**注意**:`faiss 1.7.2`は`macOS`で`Segmentation Fault: 11`を起こすので、`requirements.txt`の該当行を `faiss-cpu==1.7.0`に変更してください。

```bash
pip install -r requirements.txt
```

## 基底modelsを準備
RVCは推論/訓練のために、様々な事前訓練を行った基底modelsが必要です。

modelsは[Hugging Face space](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)からダウンロードできます。

以下は、RVCに必要な基底modelsやその他のfilesの一覧です。
```bash
hubert_base.pt

./pretrained 

./uvr5_weights

# ffmpegがすでにinstallされている場合はskip
./ffmpeg
```
その後、下記のcommandでWebUIを起動
```bash
python infer-web.py
```
Windowsをお使いの方は、直接に`RVC-beta.7z`をダウンロード後に展開し、`go-web.bat`をclickでWebUIを起動。(7zipが必要です)

また、repoに[小白简易教程.doc](./小白简易教程.doc)がありますので、参考にしてください（中国語版のみ）。

## 参考プロジェクト
+ [ContentVec](https://github.com/auspicious3000/contentvec/)
+ [VITS](https://github.com/jaywalnut310/vits)
+ [HIFIGAN](https://github.com/jik876/hifi-gan)
+ [Gradio](https://github.com/gradio-app/gradio)
+ [FFmpeg](https://github.com/FFmpeg/FFmpeg)
+ [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
+ [audio-slicer](https://github.com/openvpi/audio-slicer)

## 貢献者(contributer)の皆様の尽力に感謝します
<a href="https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=liujing04/Retrieval-based-Voice-Conversion-WebUI" />
</a>
