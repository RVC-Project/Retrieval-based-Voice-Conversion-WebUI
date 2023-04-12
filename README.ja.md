<div align="center">

<h1>Retrieval-based-Voice-Conversion-WebUI</h1>
使いやすいVITSベースの音声変換(ボイスチェンジャー)フレームワーク<br><br>

[![madewithlove](https://forthebadge.com/images/badges/built-with-love.svg)](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI)

<img src="https://counter.seku.su/cmoe?name=rvc&theme=r34" /><br>

[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/liujing04/Retrieval-based-Voice-Conversion-WebUI/blob/main/Retrieval_based_Voice_Conversion_WebUI.ipynb)
[![Licence](https://img.shields.io/github/license/liujing04/Retrieval-based-Voice-Conversion-WebUI?style=for-the-badge)](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI/blob/main/%E4%BD%BF%E7%94%A8%E9%9C%80%E9%81%B5%E5%AE%88%E7%9A%84%E5%8D%8F%E8%AE%AE-LICENSE.txt)
[![Huggingface](https://img.shields.io/badge/🤗%20-Spaces-blue.svg?style=for-the-badge)](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)

</div>

------

[**ChangeLog**](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI/blob/main/Changelog_CN.md)

[**English**](./README.en.md) | [**中文简体**](./README.md) | [**日本語**](./README.ja.md)

> [デモ映像](https://www.bilibili.com/video/BV1pm4y1z7Gm/)はこちらからご覧いただけます

> RVCによるリアルタイム音声変換: [w-okada/voice-changer](https://github.com/w-okada/voice-changer)

## はじめに
本リポジトリには以下の特徴がある：
+ top1検索を利用して、ソース特徴量をトレーニングセット特徴量に置き換えることで、トーンリークを低減する;
+ 比較的貧弱なグラフィックカードでも、簡単かつ高速にトレーニングできる;
+ 少量のデータで比較的良好な結果が得られる(10分以上の低ノイズ音声を推奨);
+ 音色を変えるためのモデルマージをサポート(ckpt processingタブ->ckpt mergeを使用);
+ 使いやすいWebuiインターフェース;
+ ボーカルと楽器を素早く分割するために、UVR5モデルを使用することができます。
+ 事前学習モデルのデータセットには、約50時間に及ぶ高品質なVCTKオープンソースデータセットが使用されており、著作権侵害を心配することなく使用できるよう、高品質なライセンス楽曲データセットが次々とトレーニングセットに追加されます。
## 環境構築
poetryで依存関係をインストールすることをお勧めします。

以下のコマンドは、Python3.8以上の環境下で実行する必要があります:
```bash
# PyTorch関連の依存関係をインストール。インストール済の場合はスキップ
# 参照先: https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio

#Windows＋ Nvidia Ampere Architecture(RTX30xx)の場合、https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI/issues/21 のissueに従い、pytorchに対応するcudaバージョンを指定する必要があります。

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# PyTorch関連の依存関係をインストール。インストール済の場合はスキップ
# 参照先: https://python-poetry.org/docs/#installation
curl -sSL https://install.python-poetry.org | python3 -

# Poetry経由で依存関係をインストール
poetry install
```

pipでも依存関係のインストールが可能です:

**注意**:`faiss 1.7.2`は`macOS`で`Segmentation Fault: 11`が発生するので、`requirements.txt`の該当行を `faiss-cpu==1.7.0`に変更してください。

```bash
pip install -r requirements.txt
```

## その他モデル前の準備
RVCは推論と訓練のために、他の多くのPre Trained Modelを必要とします。

これらのモデルは[Hugging Face space](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)から取得することが可能です。

以下は、RVCに必要なPre Trained Modelやその他のファイルの一覧です。
```bash
hubert_base.pt

./pretrained 

./uvr5_weights

# ffmpegがすでにインストールされている場合はスキップ。
./ffmpeg
```
その後、以下のコマンドでWebuiを起動
```bash
python infer-web.py
```
Windowsをお使いの方は、直接`RVC-beta.7z`をダウンロードして解凍してRVCを使い、`go-web.bat`を実行してWebUIを起動することができます。

WebUIの英語版は2週間ほどで公開する予定です。

また、リポジトリに[小白简易教程.doc](./小白简易教程.doc)がありますので、参考にしてください。

## 参考資料等
+ [ContentVec](https://github.com/auspicious3000/contentvec/)
+ [VITS](https://github.com/jaywalnut310/vits)
+ [HIFIGAN](https://github.com/jik876/hifi-gan)
+ [Gradio](https://github.com/gradio-app/gradio)
+ [FFmpeg](https://github.com/FFmpeg/FFmpeg)
+ [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
+ [audio-slicer](https://github.com/openvpi/audio-slicer)
## コントリビュータの皆様の努力に感謝します
<a href="https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=liujing04/Retrieval-based-Voice-Conversion-WebUI" />
</a>

