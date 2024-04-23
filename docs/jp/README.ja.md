<div align="center">

<h1>Retrieval-based-Voice-Conversion-WebUI</h1>
VITSに基づく使いやすい音声変換（voice changer）framework<br><br>

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

<img src="https://counter.seku.su/cmoe?name=rvc&theme=r34" /><br>

[![RVC v1](https://img.shields.io/badge/RVCv1-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/tools/ipynb/v1.ipynb)
[![RVC v2](https://img.shields.io/badge/RVCv2-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/tools/ipynb/v2.ipynb)
[![Licence](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-Spaces-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)

[![Discord](https://img.shields.io/badge/RVC%20Developers-Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/HcsmBBGyVk)

[**更新日誌**](./Changelog_JA.md) | [**よくある質問**](./faq_ja.md) | [**AutoDLで推論(中国語のみ)**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/Autodl%E8%AE%AD%E7%BB%83RVC%C2%B7AI%E6%AD%8C%E6%89%8B%E6%95%99%E7%A8%8B) | [**対照実験記録**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/%E5%AF%B9%E7%85%A7%E5%AE%9E%E9%AA%8C%C2%B7%E5%AE%9E%E9%AA%8C%E8%AE%B0%E5%BD%95) | [**オンラインデモ(中国語のみ)**](https://modelscope.cn/studios/FlowerCry/RVCv2demo)

[**English**](../en/README.en.md) | [**中文简体**](../../README.md) | [**日本語**](../jp/README.ja.md) | [**한국어**](../kr/README.ko.md) ([**韓國語**](../kr/README.ko.han.md)) | [**Français**](../fr/README.fr.md) | [**Türkçe**](../tr/README.tr.md) | [**Português**](../pt/README.pt.md)

</div>

> 著作権侵害を心配することなく使用できるよう、約 50 時間の高品質なオープンソースデータセットを使用し、基底モデルを学習し出しました。

> RVCv3 の基底モデルをご期待ください。より大きなパラメータ、より大きなデータ、より良い効果を提供し、基本的に同様の推論速度を維持しながら学習に必要なデータ量はより少なくなります。

> モデルや統合パッケージをダウンロードしやすい[RVC-Models-Downloader](https://github.com/RVC-Project/RVC-Models-Downloader)のご利用がお勧めです。

<table>
   <tr>
		<td align="center">学習・推論</td>
		<td align="center">即時音声変換</td>
	</tr>
  <tr>
		<td align="center"><img src="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/assets/129054828/092e5c12-0d49-4168-a590-0b0ef6a4f630"></td>
    <td align="center"><img src="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/assets/129054828/730b4114-8805-44a1-ab1a-04668f3c30a6"></td>
	</tr>
	<tr>
		<td align="center">go-web.bat</td>
		<td align="center">go-realtime-gui.bat</td>
	</tr>
  <tr>
    <td align="center">実行したい操作を自由に選択できます。</td>
	<td align="center">既に端から端までの170msの遅延を実現しました。ASIO入出力デバイスを使用すれば、端から端までの90msの遅延を達成できますが、ハードウェアドライバーの支援に非常に依存しています。</td>
	</tr>
</table>

## はじめに

本リポジトリには下記の特徴があります。

- Top1 検索を用いることで、生の特徴量を学習用データセット特徴量に変換し、トーンリーケージを削減します。
- 比較的貧弱な GPU でも、高速かつ簡単に学習できます。
- 少量のデータセットからでも、比較的良い結果を得ることができます。（10 分以上のノイズの少ない音声を推奨します。）
- モデルを融合することで、音声を混ぜることができます。（ckpt processing タブの、ckpt merge を使用します。）
- 使いやすい WebUI。
- UVR5 Model も含んでいるため、人の声と BGM を素早く分離できます。
- 最先端の[人間の声のピッチ抽出アルゴリズム InterSpeech2023-RMVPE](#参照プロジェクト)を使用して無声音問題を解決します。効果は最高（著しく）で、crepe_full よりも速く、リソース使用が少ないです。
- AMD GPU と Intel GPU の加速サポート

デモ動画は[こちら](https://www.bilibili.com/video/BV1pm4y1z7Gm/)でご覧ください。

## 環境構築
### Python バージョン制限
> conda で Python 環境を管理することがお勧めです

> バージョン制限の原因はこの [bug](https://github.com/facebookresearch/fairseq/issues/5012) を参照してください。

```bash
python --version # 3.8 <= Python < 3.11
```

### Linux/MacOS ワンクリック依存関係インストール・起動するスクリプト
プロジェクトのルートディレクトリで`run.sh`を実行するだけで、`venv`仮想環境を一括設定し、必要な依存関係を自動的にインストールし、メインプログラムを起動できます。
```bash
sh ./run.sh
```

### 依存関係のマニュアルインストレーション
1. `pytorch`とそのコア依存関係をインストールします。すでにインストールされている場合は見送りできます。参考: https://pytorch.org/get-started/locally/
	```bash
	pip install torch torchvision torchaudio
	```
2. もし、Windows + Nvidia Ampere (RTX30xx)の場合、#21 の経験に基づき、pytorchの対応する CUDA バージョンを指定する必要があります。
	```bash
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
	```
3. 自分の GPU に対応する依存関係をインストールします。
- Nvidia GPU
	```bash
	pip install -r requirements.txt
	```
- AMD/Intel GPU
	```bash
	pip install -r requirements-dml.txt
	```
- AMD ROCM (Linux)
	```bash
	pip install -r requirements-amd.txt
	```
- Intel IPEX (Linux)
	```bash
	pip install -r requirements-ipex.txt
	```

## その他のデータを準備

### 1. アセット
> RVCは、`assets`フォルダにある幾つかのモデルリソースで推論・学習することが必要です。
#### リソースの自動チェック/ダウンロード（デフォルト）
> デフォルトでは、RVC は主プログラムの起動時に必要なリソースの完全性を自動的にチェックしできます。

> リソースが不完全でも、プログラムは起動し続けます。

- すべてのリソースをダウンロードしたい場合は、`--update`パラメータを追加してください。
- 起動時のリソース完全性チェックを不要の場合は、`--nocheck`パラメータを追加してください。

#### リソースのマニュアルダウンロード
> すべてのリソースファイルは[Hugging Face space](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)にあります。

> `tools`フォルダでそれらをダウンロードするスクリプトを見つけることができます。

> モデル/統合パッケージ/ツールの一括ダウンローダー、[RVC-Models-Downloader](https://github.com/RVC-Project/RVC-Models-Downloader)も使用できます。

以下は、RVCが必要とするすべての事前モデルデータやその他のファイルの名前を含むリストです。

- ./assets/hubert/hubert_base.pt
	```bash
	rvcmd assets/hubert # RVC-Models-Downloader command
	```
- ./assets/pretrained
	```bash
	rvcmd assets/v1 # RVC-Models-Downloader command
	```
- ./assets/uvr5_weights
	```bash
	rvcmd assets/uvr5 # RVC-Models-Downloader command
	```
v2バージョンのモデルを使用したい場合は、追加ダウンロードが必要です。

- ./assets/pretrained_v2
	```bash
	rvcmd assets/v2 # RVC-Models-Downloader command
	```

### 2. ffmpegツールのインストール
`ffmpeg`と`ffprobe`がすでにインストールされている場合は、このステップをスキップできます。

#### Ubuntu/Debian
```bash
sudo apt install ffmpeg
```
#### MacOS
```bash
brew install ffmpeg
```
#### Windows
ダウンロード後、ルートディレクトリに配置しましょう。
```bash
rvcmd tools/ffmpeg # RVC-Models-Downloader command
```
- [ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffmpeg.exe)

- [ffprobe.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffprobe.exe)

### 3. RMVPE人声音高抽出アルゴリズムに必要なファイルのダウンロード

最新のRMVPE人声音高抽出アルゴリズムを使用したい場合は、音高抽出モデルをダウンロードし、`assets/rmvpe`に配置する必要があります。

- [rmvpe.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt)
	```bash
	rvcmd assets/rmvpe # RVC-Models-Downloader command
	```

#### RMVPE(dml環境)のダウンロード（オプション、AMD/Intel GPU ユーザー）

- [rmvpe.onnx](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.onnx)
	```bash
	rvcmd assets/rmvpe # RVC-Models-Downloader command
	```

### 4. AMD ROCM（オプション、Linuxのみ）

AMDのRocm技術を基にLinuxシステムでRVCを実行したい場合は、まず[ここ](https://rocm.docs.amd.com/en/latest/deploy/linux/os-native/install.html)で必要なドライバをインストールしてください。

Arch Linuxを使用している場合は、pacmanを使用して必要なドライバをインストールできます。
````
pacman -S rocm-hip-sdk rocm-opencl-sdk
````
一部のグラフィックカードモデルでは、以下のような環境変数を追加で設定する必要があるかもしれません（例：RX6700XT）。
````
export ROCM_PATH=/opt/rocm
export HSA_OVERRIDE_GFX_VERSION=10.3.0
````
また、現在のユーザーが`render`および`video`ユーザーグループに所属していることを確認してください。
````
sudo usermod -aG render $USERNAME
sudo usermod -aG video $USERNAME
````

## 利用開始
### 直接起動
以下のコマンドで WebUI を起動します
```bash
python infer-web.py
```
### Linux/MacOS
```bash
./run.sh
```
### IPEX 技術が必要な Intel GPU ユーザー向け(Linux のみ)
```bash
source /opt/intel/oneapi/setvars.sh
./run.sh
```
### 統合パッケージの使用 (Windowsのみ)
`RVC-beta.7z`をダウンロードして解凍し、`go-web.bat`をダブルクリック。
```bash
rvcmd packs/general/latest # RVC-Models-Downloader command
```

## 参考プロジェクト
- [ContentVec](https://github.com/auspicious3000/contentvec/)
- [VITS](https://github.com/jaywalnut310/vits)
- [HIFIGAN](https://github.com/jik876/hifi-gan)
- [Gradio](https://github.com/gradio-app/gradio)
- [FFmpeg](https://github.com/FFmpeg/FFmpeg)
- [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
- [audio-slicer](https://github.com/openvpi/audio-slicer)
- [Vocal pitch extraction:RMVPE](https://github.com/Dream-High/RMVPE)
  - 事前学習されたモデルは[yxlllc](https://github.com/yxlllc/RMVPE)と[RVC-Boss](https://github.com/RVC-Boss)によって学習され、テストされました。

## すべての貢献者の努力に感謝します

<a href="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=RVC-Project/Retrieval-based-Voice-Conversion-WebUI" />
</a>
