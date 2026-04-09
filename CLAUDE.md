# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

RVC (Retrieval-based Voice Conversion) WebUI - VITSベースの音声変換フレームワーク。Top1検索を使用して音声の音色漏れを防ぎ、少量のデータ（10分程度）でも良好な結果を得られる。

## 起動コマンド

```bash
# WebUI起動（メイン）
uv run python infer-web.py

# WebUI起動（オプション付き）
uv run python infer-web.py --port 7865 --noautoopen --dml

# リアルタイム変換GUI
uv run python tools/rvc_for_realtime.py
```

## 依存関係インストール

```bash
# uv で一括インストール（Python 3.12 + PyTorch CUDA 12.4 + 全依存パッケージ）
uv sync
```

依存関係は pyproject.toml で管理。requirements.txt は廃止済み。

## CLI推論

```bash
uv run python tools/infer_cli.py \
  --model_name MODEL.pth \
  --input_path input.wav \
  --opt_path output.wav \
  --index_path logs/MODEL/added_IVF*.index \
  --f0method rmvpe
```

## アーキテクチャ

### コアモジュール (`infer/`)

- `modules/vc/` - 音声変換メインロジック
  - `modules.py` - VCクラス（モデル読み込み、推論）
  - `pipeline.py` - 推論パイプライン（F0抽出、特徴量変換）
- `modules/train/` - トレーニング関連
  - `train.py` - 分散学習対応のトレーニングループ
  - `preprocess.py` - 音声前処理
  - `extract/` - F0特徴量抽出
- `modules/uvr5/` - Ultimate Vocal Remover（ボーカル分離）
- `lib/infer_pack/` - 推論用モデル定義
  - `models.py` - SynthesizerTrnMs768NSFsid, SynthesizerTrnMs768NSFsid_nono
- `lib/rmvpe.py` - RMVPE F0抽出アルゴリズム

### 設定 (`configs/`)

- `config.py` - デバイス自動検出、精度設定（fp16/fp32）
- `v2/` - モデル設定JSON (32k, 48k)

### エントリーポイント

- `infer-web.py` - Gradio WebUI（トレーニング・推論・モデル管理）
- `api_240604.py` - FastAPI REST API

### 必要なアセット

- `assets/hubert/hubert_base.pt` - HuBERT特徴量抽出モデル
- `assets/pretrained_v2/` - 事前学習済みモデル
- `assets/uvr5_weights/` - UVR5モデル
- `assets/rmvpe/rmvpe.pt` - RMVPE F0抽出モデル

## 環境変数 (.env)

```
weight_root = assets/weights      # モデル保存先
index_root = logs                 # インデックス保存先
rmvpe_root = assets/rmvpe         # RMVPEモデル
```

## モデル仕様

- 768次元HuBERT特徴量 (32k/48kHz)
- HuBERT出力層: layer 12
- 改良版ディスクリミネータ (MultiPeriodDiscriminatorV2)

## F0抽出メソッド

- `rmvpe` - 推奨。高品質かつ高速
- `harvest` - PyWorld。安定だが低速
- `crepe` - ニューラルネット。GPU推奨
- `pm` - Parselmouth。最軽量

## コントリビューションルール

アルゴリズム変更は基本的に受け付けていない。翻訳やWebUIの改善は最小限の変更で。
