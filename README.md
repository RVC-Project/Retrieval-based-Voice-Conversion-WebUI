<div align="center">

<h1>Retrieval-based-Voice-Conversion-WebUI</h1>
VITSに基づく使いやすい音声変換（Voice Changer）フレームワーク<br><br>

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

<img src="https://counter.seku.su/cmoe?name=rvc&theme=r34" /><br>

[![Licence](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-Spaces-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)

[![Discord](https://img.shields.io/badge/RVC%20Developers-Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/HcsmBBGyVk)

[**更新履歴**](./docs/jp/Changelog_JA.md) | [**よくある質問**](./docs/jp/faq_ja.md) | [**English**](./docs/en/README.en.md) | [**中文**](./docs/cn/)

</div>

> 著作権侵害を心配することなく使用できるように、基底モデルは約50時間の高品質なオープンソースデータセット(VCTK)で訓練されています。

> RVCv3の基底モデルをご期待ください。より大きなパラメータ、より大きなデータ、より良い効果を提供し、基本的に同様の推論速度を維持しながら、トレーニングに必要なデータ量はより少なくなります。

<table>
   <tr>
		<td align="center">トレーニングと推論インターフェース</td>
		<td align="center">リアルタイム音声変換インターフェース</td>
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
		<td align="center">端から端まで170msの遅延を実現。ASIO入出力デバイスを使用すれば90msの遅延を達成可能（ハードウェア依存）。</td>
	</tr>
</table>

## はじめに

本リポジトリには下記の特徴があります。

- Top1検索を用いることで、生の特徴量を訓練用データセット特徴量に変換し、トーンリーケージを削減
- 比較的貧弱なGPUでも、高速かつ簡単に訓練可能
- 少量のデータセットからでも比較的良い結果を取得可能（10分以上のノイズの少ない音声を推奨）
- モデル融合で音声を混合可能（ckpt processingタブのckpt mergeを使用）
- 使いやすいWebUI
- UVR5 Modelで人声とBGMを素早く分離
- 最先端の[人間の声のピッチ抽出アルゴリズム InterSpeech2023-RMVPE](#参照プロジェクト)を使用
- AMD/Intel GPU対応

## 環境構築 (Windows + uv)

### 必要条件

- Windows 10/11
- Python 3.10
- NVIDIA GPU (CUDA 12.x対応)
- [uv](https://docs.astral.sh/uv/) (Pythonパッケージマネージャー)

### Step 1: uvのインストール

PowerShellで以下を実行:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

または winget を使用:

```bash
winget install --id=astral-sh.uv -e
```

### Step 2: Python 3.10環境の作成

```bash
cd Retrieval-based-Voice-Conversion-WebUI
uv venv --python 3.10
```

### Step 3: 仮想環境の有効化

```bash
.venv\Scripts\activate
```

### Step 4: PyTorchのインストール (CUDA 12.4対応)

```bash
uv pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
```

> 古いGPUやドライバーの場合は、CUDA 11.8版を使用:
> ```bash
> uv pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
> ```

### Step 5: 依存パッケージのインストール

```bash
uv pip install -r requirements.txt
```

### Step 6: fairseqのインストール

fairseqはC++コンパイルが必要なため、以下の方法でインストール:

```bash
# 方法1: 環境変数でC++拡張をスキップ
set FAIRSEQ_BUILD_EXT=0
uv pip install fairseq==0.12.2

# 方法2: Python 3.11の場合
uv pip install fairseq @ git+https://github.com/One-sixth/fairseq.git
```

### Step 7: 必要なモデルのダウンロード

```bash
python tools/download_models.py
```

または手動で[Hugging Face](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)からダウンロード:

- `assets/hubert/hubert_base.pt`
- `assets/pretrained_v2/` (事前学習モデル)
- `assets/uvr5_weights/` (UVR5モデル)
- `assets/rmvpe/rmvpe.pt` (RMVPE F0抽出モデル)

### Step 8: ffmpegの確認

ffmpegがPATHに含まれているか確認:

```bash
ffmpeg -version
```

インストールされていない場合は [ffmpeg公式サイト](https://ffmpeg.org/download.html) からダウンロードしてPATHに追加するか、プロジェクトルートに `ffmpeg.exe` と `ffprobe.exe` を配置。

## 使用方法

### WebUIの起動

```bash
# 仮想環境の有効化
.venv\Scripts\activate

# WebUI起動
python infer-web.py
```

ブラウザで http://localhost:7865 にアクセス

または `go-web.bat` をダブルクリック

### コマンドラインオプション

```bash
python infer-web.py --port 7865 --noautoopen
```

### リアルタイム音声変換

```bash
python gui_v1.py
```

または `go-realtime-gui.bat` をダブルクリック

### CLI推論

```bash
python tools/infer_cli.py ^
  --model_name MODEL.pth ^
  --input_path input.wav ^
  --opt_path output.wav ^
  --index_path logs/MODEL/added_IVF*.index ^
  --f0method rmvpe
```

## モデルの配置

### 音声モデル (.pth)

`assets/weights/` フォルダに配置

### インデックスファイル (.index)

`logs/` フォルダに配置

## トラブルシューティング

### fairseqインストールエラー

Visual Studio Build Toolsが必要な場合:

```bash
winget install Microsoft.VisualStudio.2022.BuildTools
```

または環境変数でC++拡張をスキップ:

```bash
set FAIRSEQ_BUILD_EXT=0
uv pip install fairseq==0.12.2
```

### gradio-clientエラー

`ImportError: cannot import name 'media_data' from 'gradio_client'` エラーの場合:

```bash
uv pip install gradio-client==0.2.7
```

### CUDAバージョンの確認

```bash
nvidia-smi
```

## 参照プロジェクト

- [ContentVec](https://github.com/auspicious3000/contentvec/)
- [VITS](https://github.com/jaywalnut310/vits)
- [HIFIGAN](https://github.com/jik876/hifi-gan)
- [Gradio](https://github.com/gradio-app/gradio)
- [FFmpeg](https://github.com/FFmpeg/FFmpeg)
- [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
- [audio-slicer](https://github.com/openvpi/audio-slicer)
- [Vocal pitch extraction:RMVPE](https://github.com/Dream-High/RMVPE)
  - 事前訓練されたモデルは[yxlllc](https://github.com/yxlllc/RMVPE)と[RVC-Boss](https://github.com/RVC-Boss)によって訓練され、テストされました。

## すべての貢献者の努力に感謝します

<a href="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=RVC-Project/Retrieval-based-Voice-Conversion-WebUI" />
</a>
