# M2-B: 日本語歌声事前学習

## メタ情報
- **マイルストーン**: M2
- **フェーズ**: Phase 2-B（Week 4-5）
- **工数見積もり**: 8人日 + GPU 14-30h
- **GPU要件**: RTX 4090 24GB（事前学習）、Cloud GPU（必要時）
- **前提タスク**: M2-A完了（SSLモデル統合済み）、タスク2-4完了（学習パイプラインSSL対応済み）
- **ステータス**: 未着手
- **関連マイルストーン**: [milestones.md](../milestones.md) > M2 Phase 2-Bセクション
- **追加パッケージ**: `librosa>=0.10.0`（ピッチシフト・タイムストレッチ用、pyproject.toml に既存）

---

## 1. タスク目的とゴール

### 目的

日本語歌声データセットを用いた段階的事前学習により、ターゲット話者ファインチューニングの収束速度と品質を大幅に向上させる。現行RVCの事前学習モデル（`assets/pretrained_v2/`）は英語・中国語の話し声データで学習されており、日本語歌声のドメインとは乖離がある。本タスクでは以下の2段階の転移学習パイプラインを構築する。

1. **Stage 1（多話者基盤学習）**: JVS-MuSiC 100話者の歌声データで多話者事前学習を実施し、日本語歌声の基本的な音響特徴（フォルマント構造、母音遷移、子音調音）をモデルに獲得させる
2. **Stage 2（歌声DB適応）**: NIT-SONG070、東北きりたん、東北イタコ、GTSinger(JA) の高品質歌声データで追加学習し、プロ歌唱のビブラート・ファルセット・ブレスコントロールなどの表現力を獲得させる

加えて、データ拡張パイプライン（ピッチシフト / タイムストレッチ / SpecAugment）を構築し、限られた歌声データの実効量を4-10倍に拡大する。

### なぜ必要か

1. **ドメインギャップの解消**: 現行の事前学習モデルは話し声ベースであり、歌声特有のロングトーン・ビブラート・広い音域に対する表現力が不足している。日本語歌声で事前学習することで、ファインチューニング開始時点のモデル品質が飛躍的に向上する
2. **収束速度の改善**: 適切な事前学習により、ターゲット話者FTが現行の50-100エポックから3-5エポックへ短縮されることが期待できる。これはエンドユーザーの学習時間を大幅に削減する
3. **少量データでの汎化**: 10分程度の声優歌声データから高品質モデルを構築するには、事前学習で獲得した日本語歌声の事前知識が不可欠。データ拡張との組み合わせにより過学習を防止する
4. **SSLモデルの効果最大化**: M2-Aで統合したkushinada/ContentVecの効果は、対応する事前学習モデルがあって初めて最大限発揮される

### 成功条件（Go/No-Go判定 Week 5終了時）

| 基準 | Go条件 | No-Go時の対応 |
|------|--------|--------------|
| CER改善 | kushinada/rinnaがContentVec比で**CER 5pt以上改善** | rinnaチェックポイント(BOOTH `f0X48k768_jphubert_v2`)で代替 |
| 事前学習効果 | FT収束が**3-5エポック**に短縮（現行50-100エポック比） | エポック数調整、学習率変更、Stage 2データ追加 |
| 歌声品質 | 主観評価でベースライン（ContentVec + 既存pretrained_v2）以上 | SSLモデル候補の再検討（rinna, Spin V2） |

### 本タスクが生成する成果物

| 成果物 | パス | 説明 |
|--------|------|------|
| データ拡張スクリプト | `tools/augment/pitch_shift.py` | ピッチシフト・タイムストレッチ・SpecAugment適用 |
| 多話者filelist生成スクリプト | `tools/pretrain/gen_filelist.py` | 複数データセットから統一filelistを生成 |
| 事前学習用config | `configs/v2/48k_pretrain.json` | 多話者事前学習専用の設定ファイル |
| Stage 1事前学習モデル | `assets/pretrained_v2/jpn_singing_s1_G.pth` / `D.pth` | JVS-MuSiC 100話者事前学習済み |
| Stage 2事前学習モデル | `assets/pretrained_v2/jpn_singing_s2_G.pth` / `D.pth` | 歌声DB適応済み |
| kushinada版バリアント | `assets/pretrained_v2/jpn_singing_kushinada_G.pth` / `D.pth` | kushinada SSLモデル版 |
| ContentVec版バリアント | `assets/pretrained_v2/jpn_singing_contentvec_G.pth` / `D.pth` | ContentVec SSLモデル版（比較用） |

---

## 2. 実装する内容の詳細

### サブタスク一覧

- [ ] **2-7**: データセット準備（JVS-MuSiC 24k->48kリサンプル等）
- [ ] **2-7b**: データ拡張スクリプト（ピッチシフト+-4半音）
- [ ] **2-8**: 多話者filelist生成スクリプト
- [ ] **2-9**: Stage 1: JVS-MuSiC 100話者事前学習
- [ ] **2-10**: Stage 2: 歌声DB適応（NIT+きりたん+GTSinger）
- [ ] **2-11**: ContentVec版 + kushinada版の2バリアント作成
- [ ] **2-12**: ターゲット話者ファインチューニング + 品質評価

---

### 2-7: データセット準備（2日）

#### 概要

5つの日本語歌声データセットをダウンロードし、RVCの学習パイプラインが消費できる形式（48kHz mono WAV + 16kHz resample）に統一する。

#### 対象データセット

| データセット | 話者数 | 時間 | 元SR | ライセンス | Stage | 入手方法 |
|-------------|--------|------|------|-----------|-------|---------|
| JVS-MuSiC | 100 | ~3.3h | 24kHz | 商用可 | 1 | [高道研究室](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_music) からダウンロード |
| NIT-SONG070 | 1 | 1.2h | 48kHz | CC BY 3.0 | 2 | [SourceForge](https://sourceforge.net/projects/sinsy/files/) |
| GTSinger(JA) | 2 | ~8h | 48kHz/24bit | 研究OS | 2 | [HuggingFace](https://huggingface.co/datasets/GTSinger/GTSinger) |
| 東北きりたん | 1 | 57min | 48kHz | 研究のみ | 2 | [zunko.jp](https://zunko.jp/kiridev/login.php) 要ユーザ登録 |
| 東北イタコ | 1 | ~1h | 48kHz | 研究のみ | 2 | [zunko.jp](https://zunko.jp/itadev/login.php) 要ユーザ登録 |

#### 処理手順

**Step 1: ディレクトリ構造の作成**

```
datasets/
  pretrain/
    jvs_music/          # JVS-MuSiC（100話者分のサブディレクトリ）
      spk001/
      spk002/
      ...
      spk100/
    nit_song070/        # NIT-SONG070
    gtsinger_ja/        # GTSinger日本語部分
      JA-Soprano-1/
      JA-Tenor-1/
    kiritan/            # 東北きりたん
    itako/              # 東北イタコ
```

**Step 2: JVS-MuSiC 24kHz -> 48kHz リサンプリング**

JVS-MuSiCは24kHz収録のため、48kHzへのアップサンプリングが必要。ただし**24kHzの信号には12kHz以上の高域成分が存在しない**ため、以下の対策を取る。

```python
import librosa
import soundfile as sf

# 高品質リサンプリング（soxr_hqバックエンド）
y, sr = librosa.load(input_path, sr=24000, mono=True)
y_48k = librosa.resample(y, orig_sr=24000, target_sr=48000, res_type='soxr_hq')
sf.write(output_path, y_48k, 48000, subtype='FLOAT')
```

**高域欠損への対策**:
- Stage 1事前学習の設定で `mel_fmax=12000` に制限する（`configs/v2/48k_pretrain.json`）。12kHz以上にエネルギーがないため、その領域のmel損失が学習を妨害するのを防ぐ
- 代替案: 32kHz configで事前学習する場合は `configs/v2/32k.json` を使用（`sampling_rate: 32000`、Nyquist=16kHz）

**Step 3: 48kHzデータセットの前処理**

NIT-SONG070, GTSinger, きりたん, イタコは元が48kHzなのでリサンプリング不要。ただし以下を確認する:

- モノラル化（ステレオの場合は `librosa.to_mono()` でダウンミックス）
- GTSingerの24bit -> float32変換（`soundfile` が自動処理）
- 無音トリミング（先頭・末尾の2秒以上の無音を除去）
- ファイル名の正規化（日本語ファイル名をASCIIに変換、パイプ文字 `|` を含まないこと）

**Step 4: RVC前処理パイプライン適用**

各データセットに対して既存の前処理を実行:

```bash
# 話者ごとに実行（JVS-MuSiCの場合は100回）
uv run python infer/modules/train/preprocess.py \
  datasets/pretrain/jvs_music/spk001 \
  48000 \
  4 \
  logs/pretrain_jpn_singing/spk001 \
  False \
  3.7
```

これにより各話者ディレクトリに以下が生成される:
- `0_gt_wavs/` -- 48kHzセグメント（`preprocess.py` L53: `self.gt_wavs_dir`）
- `1_16k_wavs/` -- 16kHzリサンプル済みセグメント（`preprocess.py` L54: `self.wavs16k_dir`）

**Step 5: F0抽出 + SSL特徴量抽出**

```bash
# F0抽出（rmvpe推奨）
uv run python infer/modules/train/extract/extract_f0_print.py \
  logs/pretrain_jpn_singing 4 rmvpe

# SSL特徴量抽出（M2-Aでkushinada対応済みの場合はkushinadaモデルを使用）
uv run python infer/modules/train/extract_feature_print.py \
  cuda 4 0 logs/pretrain_jpn_singing v2 True
```

出力:
- `2_f0/` -- F0ピッチ（`.npy`）
- `2_f0nsf/` -- F0連続値（`.npy`）
- `3_feature768/` -- SSL特徴量768次元（`.npy`）。現行は `extract_feature_print.py` L59 `outPath = "%s/3_feature768"` で固定

#### 注意点

- JVS-MuSiCの利用規約では「再配布不可」「公開アップロード3ファイルまで」。データセット自体をgitリポジトリに含めないこと
- GTSingerは全体80.59hのうち日本語部分（JA-Soprano-1, JA-Tenor-1）のみ使用。`filter` でJAプレフィックスのサブセットを抽出する
- きりたん・イタコはユーザ登録が必要。自動ダウンロードスクリプトは作成しない（利用規約上、手動ダウンロードが推奨）

---

### 2-7b: データ拡張スクリプト（1.5日）

#### 概要

ピッチシフトとタイムストレッチによるオフラインデータ拡張スクリプトを新規作成する。拡張データはRVC前処理パイプライン（preprocess.py -> F0抽出 -> SSL特徴量抽出）の**前**に適用し、拡張後の音声に対して改めて特徴量を抽出する。

#### 新規ファイル: `tools/augment/pitch_shift.py`

```python
"""
データ拡張スクリプト: ピッチシフト + タイムストレッチ

使用例:
  uv run python tools/augment/pitch_shift.py \
    --input_dir datasets/pretrain/kiritan \
    --output_dir datasets/pretrain/kiritan_augmented \
    --shifts -4 -2 2 4 \
    --stretches 0.9 1.1 \
    --sr 48000
"""
import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def pitch_shift_file(input_path, output_path, n_steps, sr):
    """ピッチシフトを適用して保存"""
    y, _ = librosa.load(input_path, sr=sr, mono=True)
    y_shifted = librosa.effects.pitch_shift(
        y, sr=sr, n_steps=n_steps,
        bins_per_octave=12,
        res_type='soxr_hq'
    )
    sf.write(output_path, y_shifted, sr, subtype='FLOAT')


def time_stretch_file(input_path, output_path, rate, sr):
    """タイムストレッチを適用して保存"""
    y, _ = librosa.load(input_path, sr=sr, mono=True)
    y_stretched = librosa.effects.time_stretch(y, rate=rate)
    sf.write(output_path, y_stretched, sr, subtype='FLOAT')


def main():
    parser = argparse.ArgumentParser(description="歌声データ拡張")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--shifts", type=float, nargs="+", default=[-4, -2, 2, 4])
    parser.add_argument("--stretches", type=float, nargs="+", default=[0.9, 1.1])
    parser.add_argument("--sr", type=int, default=48000)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 原音をコピー
    wav_files = sorted(input_dir.glob("**/*.wav"))
    print(f"Found {len(wav_files)} WAV files")

    for wav_path in wav_files:
        rel = wav_path.relative_to(input_dir)
        # 原音コピー
        orig_out = output_dir / rel
        orig_out.parent.mkdir(parents=True, exist_ok=True)
        if not (args.skip_existing and orig_out.exists()):
            y, _ = librosa.load(str(wav_path), sr=args.sr, mono=True)
            sf.write(str(orig_out), y, args.sr, subtype='FLOAT')

        # ピッチシフト
        for n_steps in args.shifts:
            suffix = f"_ps{n_steps:+.0f}"
            shifted_out = output_dir / rel.parent / f"{rel.stem}{suffix}.wav"
            if args.skip_existing and shifted_out.exists():
                continue
            pitch_shift_file(str(wav_path), str(shifted_out), n_steps, args.sr)

        # タイムストレッチ
        for rate in args.stretches:
            suffix = f"_ts{rate:.1f}"
            stretched_out = output_dir / rel.parent / f"{rel.stem}{suffix}.wav"
            if args.skip_existing and stretched_out.exists():
                continue
            time_stretch_file(str(wav_path), str(stretched_out), rate, args.sr)

    print(f"Augmentation complete. Output: {output_dir}")


if __name__ == "__main__":
    main()
```

#### 拡張パラメータ

| 拡張種別 | パラメータ | 倍率 | 根拠 |
|---------|-----------|------|------|
| ピッチシフト | -4, -2, +2, +4 半音 | 4倍 | SPA-SVC (Interspeech 2024) で有効性実証。6半音超はフォルマント破綻 |
| タイムストレッチ | 0.9x, 1.1x | 2倍 | 10%以内はアーティファクト知覚困難 |
| SpecAugment | `FrequencyMasking(20)`, `TimeMasking(80)` | - | 学習時オンライン適用（`data_utils.py`に追加） |

**合計拡張倍率**: 原音1 + ピッチ4 + ストレッチ2 = **7倍**（10分 -> 70分相当）

#### SpecAugmentの学習時オンライン適用

SpecAugmentはオフライン拡張ではなく、`infer/lib/train/data_utils.py` の `TextAudioLoaderMultiNSFsid.get_audio()` メソッド（L98-134）にオンラインで追加する:

```python
# data_utils.py TextAudioLoaderMultiNSFsid.get_audio() 内
# spec 計算後に追加
if self.training and hasattr(self, 'spec_augment') and self.spec_augment:
    import torchaudio.transforms as T
    freq_mask = T.FrequencyMasking(freq_mask_param=20)
    time_mask = T.TimeMasking(time_mask_param=80)
    spec = freq_mask(spec)
    spec = time_mask(spec)
```

#### 注意事項

- ピッチシフト・タイムストレッチは前処理（`preprocess.py`）の**入力**に対して適用する。拡張後の音声に対してF0とSSL特徴量を改めて抽出することで、F0値と特徴量の整合性を保証する
- +-4半音を超えるシフトはフォルマント構造を破壊するため使用しない
- タイムストレッチ0.8x未満はピッチ品質が低下するため使用しない
- 拡張データのファイル名にサフィックス（`_ps+2`, `_ts0.9`等）を付けて原音と区別する

---

### 2-8: 多話者filelist生成スクリプト（1.5日）

#### 概要

複数データセットのRVC前処理済み出力を統合し、多話者学習用の `filelist.txt` を生成するスクリプトを新規作成する。

#### filelistフォーマット

RVCのfilelistは `|` 区切りのテキストファイル。`infer/lib/train/utils.py` L269 の `load_filepaths_and_text()` で読み込まれ、`data_utils.py` L34 の `TextAudioLoaderMultiNSFsid._filter()` 内のL43で以下のカラムとしてパースされる:

```
audio_path|phone_feature_path|pitch_path|pitchf_path|speaker_id
```

具体例:
```
logs/pretrain/spk001/0_gt_wavs/001_0.wav|logs/pretrain/spk001/3_feature768/001_0.npy|logs/pretrain/spk001/2_f0/001_0.wav.npy|logs/pretrain/spk001/2_f0nsf/001_0.wav.npy|0
logs/pretrain/spk001/0_gt_wavs/001_1.wav|logs/pretrain/spk001/3_feature768/001_1.npy|logs/pretrain/spk001/2_f0/001_1.wav.npy|logs/pretrain/spk001/2_f0nsf/001_1.wav.npy|0
logs/pretrain/spk002/0_gt_wavs/002_0.wav|logs/pretrain/spk002/3_feature768/002_0.npy|logs/pretrain/spk002/2_f0/002_0.wav.npy|logs/pretrain/spk002/2_f0nsf/002_0.wav.npy|1
```

#### 新規ファイル: `tools/pretrain/gen_filelist.py`

```python
"""
多話者filelist生成スクリプト

使用例:
  uv run python tools/pretrain/gen_filelist.py \
    --exp_dirs logs/pretrain/spk001 logs/pretrain/spk002 ... \
    --output logs/pretrain_jpn_singing/filelist.txt \
    --verify
"""
import argparse
import os
from pathlib import Path


def generate_filelist(exp_dirs, output_path, verify=False):
    """複数の実験ディレクトリからfilelistを生成"""
    entries = []
    missing = []

    for spk_id, exp_dir in enumerate(exp_dirs):
        exp_dir = Path(exp_dir)
        gt_dir = exp_dir / "0_gt_wavs"
        feat_dir = exp_dir / "3_feature768"
        f0_dir = exp_dir / "2_f0"
        f0nsf_dir = exp_dir / "2_f0nsf"

        if not gt_dir.exists():
            print(f"WARN: {gt_dir} does not exist, skipping")
            continue

        for wav_file in sorted(gt_dir.glob("*.wav")):
            stem = wav_file.stem
            feat_path = feat_dir / f"{stem}.npy"
            f0_path = f0_dir / f"{stem}.wav.npy"
            f0nsf_path = f0nsf_dir / f"{stem}.wav.npy"

            if verify:
                for p in [feat_path, f0_path, f0nsf_path]:
                    if not p.exists():
                        missing.append(str(p))
                        continue

            entry = f"{wav_file}|{feat_path}|{f0_path}|{f0nsf_path}|{spk_id}"
            entries.append(entry)

    if missing:
        print(f"WARNING: {len(missing)} missing files")
        for m in missing[:10]:
            print(f"  {m}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(entry + "\n")

    print(f"Generated filelist: {len(entries)} entries, {len(exp_dirs)} speakers -> {output}")
    return entries
```

#### spk_embed_dim の変更

現行の `configs/v2/48k.json` では `spk_embed_dim: 109` となっている（L44）。これは元のRVC事前学習データの話者数に基づく値。多話者事前学習ではデータセット構成に応じてこの値を変更する必要がある。

| Stage | 話者数 | spk_embed_dim |
|-------|--------|---------------|
| Stage 1 | 100 (JVS-MuSiC) | 256 |
| Stage 2 | 105 (Stage 1 + NIT + GTSinger(2) + きりたん + イタコ) | 256 |
| ターゲットFT | 1 | 1 |

`spk_embed_dim` は `nn.Embedding(spk_embed_dim, gin_channels)` として使用される（`models.py` L641: `self.emb_g = nn.Embedding(self.spk_embed_dim, gin_channels)`）。事前学習で256に拡大しても、FT時にspk_embed_dim=1に縮小するため、weight形状の不一致が発生する。

**注意: 2つの異なるロードパスが存在する**:
- **チェックポイント再開時**（`utils.py` `load_checkpoint()` L99-127）: shape不一致キーを自動で現モデルの値にフォールバックし、`strict=False` でロードするため安全に動作する
- **事前学習初回ロード時**（`train.py` L215-227）: `load_state_dict()` に `strict=False` が**付いていない**ため、spk_embed_dimが異なるとエラーになる。**M2-Bでコード修正が必要**（後述の懸念事項#2を参照）

Stage 1 -> Stage 2間ではspk_embed_dimを同じ256に維持することでこの問題を回避する。事前学習モデル -> ターゲットFT間ではspk_embed_dimが変わるため、`train.py` への `strict=False` 追加が前提となる。

#### 事前学習用config: `configs/v2/48k_pretrain.json`

既存の `configs/v2/48k.json` を複製し、以下を変更:

```json
{
  "train": {
    "log_interval": 200,
    "seed": 1234,
    "epochs": 200,
    "learning_rate": 2e-4,
    "betas": [0.8, 0.99],
    "eps": 1e-9,
    "batch_size": 8,
    "fp16_run": true,
    "lr_decay": 0.999,
    "segment_size": 17280,
    "init_lr_ratio": 1,
    "warmup_epochs": 5,
    "c_mel": 45,
    "c_kl": 1.0
  },
  "data": {
    "max_wav_value": 32768.0,
    "sampling_rate": 48000,
    "filter_length": 2048,
    "hop_length": 480,
    "win_length": 2048,
    "n_mel_channels": 128,
    "mel_fmin": 0.0,
    "mel_fmax": 12000
  },
  "model": {
    "inter_channels": 192,
    "hidden_channels": 192,
    "filter_channels": 768,
    "n_heads": 2,
    "n_layers": 6,
    "kernel_size": 3,
    "p_dropout": 0.1,
    "resblock": "1",
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "upsample_rates": [12, 10, 2, 2],
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [24, 20, 4, 4],
    "use_spectral_norm": false,
    "gin_channels": 256,
    "spk_embed_dim": 256
  }
}
```

**既存configとの差分**:

| パラメータ | 既存 `48k.json` | 事前学習用 `48k_pretrain.json` | 変更理由 |
|-----------|----------------|-------------------------------|---------|
| `epochs` | 20000 | 200 | 事前学習は200エポックで十分 |
| `learning_rate` | 1e-4 | 2e-4 | 多話者データでは高めのLRで収束促進 |
| `batch_size` | 4 | 8 | RTX 4090 24GBならbatch_size=8が可能 |
| `lr_decay` | 0.999875 | 0.999 | 200エポックに合わせて減衰を調整 |
| `warmup_epochs` | 0 | 5 | 事前学習初期の安定性確保 |
| `p_dropout` | 0 | 0.1 | 多話者での過学習防止（M1-4で変更予定の値） |
| `mel_fmax` | null (=24000) | 12000 | JVS-MuSiC 24kHz由来データの高域欠損対策 |
| `spk_embed_dim` | 109 | 256 | 100話者 + 余裕分 |

---

### 2-9: Stage 1 JVS-MuSiC 100話者事前学習（1日 / GPU ~4h）

#### 概要

JVS-MuSiC 100話者のデータ（~3.3h、拡張後~23h）を用いて、日本語歌声の基礎的な音響モデルを事前学習する。

#### 実行手順

```bash
# 1. config配置
cp configs/v2/48k_pretrain.json logs/pretrain_jpn_singing_s1/config.json

# 2. 学習実行
uv run python infer/modules/train/train.py \
  -se 10 \
  -te 200 \
  -pg "" \
  -pd "" \
  -g 0 \
  -bs 8 \
  -e pretrain_jpn_singing_s1 \
  -sr 48k \
  -sw 1 \
  -v v2 \
  -f0 1 \
  -l 0 \
  -c 0
```

#### 学習パラメータ

| パラメータ | 値 | 根拠 |
|-----------|-----|------|
| 総エポック | 200 | 3.3h(raw) * 7(拡張) = ~23hデータ、200エポックで十分な収束 |
| バッチサイズ | 8 | RTX 4090 24GB VRAM。fp16使用 |
| 学習率 | 2e-4 | 多話者の多様性を捕捉するため標準より高め |
| 保存間隔 | 10エポック | 学習曲線の確認用 |
| pretrainG/D | なし（ランダム初期化） | 英語pretrained_v2からの転移は逆効果の可能性あり |
| mel_fmax | 12000 | JVS-MuSiC 24kHzのNyquist制限 |

#### VRAM見積もり

```
モデルパラメータ: ~56M params (SynthesizerTrnMs768NSFsid)
fp16メモリ: ~112MB (params) + ~112MB (gradients) + ~200MB (optimizer states)
バッチデータ: batch_size=8, segment_size=17280 -> ~1.5GB
ディスクリミネータ: ~20M params -> ~300MB
合計推定: ~8-12GB (24GB VRAM内に収まる)
```

バッチサイズ8でOOMが発生する場合は6->4と段階的に下げる。batch_size=4でも~6GBで収まるため、RTX 4070 Ti Super 16GBでも実行可能。

#### チェックポイント

- `logs/pretrain_jpn_singing_s1/G_*.pth` / `D_*.pth` がエポックごとに保存される
- TensorBoardで `loss/g/total`, `loss/d/total`, `loss/g/mel`, `loss/g/kl` の推移を監視
- 目安: 100エポック時点で `loss/g/mel < 30`, `loss/g/kl < 3` 程度に収束

---

### 2-10: Stage 2 歌声DB適応（1日 / GPU ~10h）

#### 概要

Stage 1モデルを初期値として、高品質歌声データベース（NIT-SONG070, きりたん, イタコ, GTSinger(JA)）で追加学習する。Stage 2ではプロ歌唱の表現力（ビブラート、ファルセット、ブレスコントロール、J-POPの楽曲構造）を獲得させる。

#### データセット詳細

| データセット | 特徴 | 追加学習での役割 |
|-------------|------|----------------|
| NIT-SONG070 | 48kHz, CC BY 3.0, 女性1名, 童謡 | 高品質48kHz基盤（mel_fmax制限を解除可能） |
| 東北きりたん | 48kHz, 研究のみ, 女性1名, **J-POP** | J-POPジャンルの楽曲構造・発声パターン |
| 東北イタコ | 48kHz, 研究のみ, 女性1名, J-POP | きりたんの姉妹プロジェクト、追加のJ-POP |
| GTSinger(JA) | 48kHz/24bit, 研究OS, ソプラノ+テナー | 6歌唱技法（ミックスボイス、ファルセット、ビブラート等） |

#### Stage 2のconfig変更

`configs/v2/48k_pretrain.json` のコピーを `48k_pretrain_s2.json` として作成し、以下を変更:

```json
{
  "data": {
    "mel_fmax": null
  },
  "train": {
    "epochs": 300,
    "learning_rate": 5e-5,
    "warmup_epochs": 3
  },
  "model": {
    "spk_embed_dim": 256
  }
}
```

- `mel_fmax: null` に戻す（Stage 2データは48kHzネイティブなので全帯域使用可能）
- 学習率を5e-5に下げる（Stage 1の重みを大きく壊さないため）
- Stage 1の話者ID（0-99）に追加して、Stage 2話者のIDを100番台に割り当てる

#### 実行手順

```bash
# Stage 1のチェックポイントからロード
uv run python infer/modules/train/train.py \
  -se 10 \
  -te 300 \
  -pg logs/pretrain_jpn_singing_s1/G_最終.pth \
  -pd logs/pretrain_jpn_singing_s1/D_最終.pth \
  -g 0 \
  -bs 6 \
  -e pretrain_jpn_singing_s2 \
  -sr 48k \
  -sw 1 \
  -v v2 \
  -f0 1 \
  -l 0 \
  -c 0
```

**注意**: `train.py` L215-227 の事前学習ロードロジックにより、`-pg` / `-pd` で指定したチェックポイントの `model` weightが `load_state_dict()` でロードされる。オプティマイザの状態はロードされない（L223のコメント参照: `测试不加载优化器`）。現行コードでは `load_state_dict()` に `strict=False` が**付いていない**ため、spk_embed_dimが異なるとRuntimeErrorが発生する。したがって、**spk_embed_dimをStage 1と同じ256に維持する**ことが重要。Stage 2の追加話者はID 100-104 に割り当てる（NIT:100, GTSinger-Soprano:101, GTSinger-Tenor:102, きりたん:103, イタコ:104）。

なお、事前学習モデル -> ターゲットFT（spk_embed_dim=1）でロードする場合は、`train.py` の該当箇所に `strict=False` を追加するコード修正が必要（懸念事項#2を参照）。代替案として `utils.py` L99-127 の `load_checkpoint()` 関数は shape不一致キーを自動でフォールバックする安全な実装になっているため、事前学習ロード時にもこの関数を呼ぶようリファクタリングする方法がある。

---

### 2-11: ContentVec版 + kushinada版の2バリアント作成（1.5日 / GPU ~14h）

#### 概要

M2-Aで統合したSSLモデル抽象化レイヤーを活用し、以下の2バリアントの事前学習モデルを作成する:

| バリアント | SSLモデル | 用途 |
|-----------|----------|------|
| **ContentVec版** | `assets/hubert/hubert_base.pt` (ContentVec) | ベースライン比較。既存互換 |
| **kushinada版** | `imprt/kushinada-hubert-base` (HuggingFace) | 日本語最適化。メイン候補 |

#### 手順

1. **SSL特徴量の再抽出**: 同一の前処理済みデータセットに対して、2つのSSLモデルでそれぞれ `3_feature768/` を再生成する
   - ContentVec版: 既存の `extract_feature_print.py` をそのまま使用（`assets/hubert/hubert_base.pt`）
   - kushinada版: M2-Aで実装したSSLローダー抽象化（タスク2-1, 2-2）を使用

2. **filelistの生成**: 特徴量パスを各バリアント用に書き換えたfilelistを生成

3. **事前学習の実行**: Stage 1 -> Stage 2 を2回（計4学習ジョブ）実行

#### GPU時間見積もり

| ジョブ | 所要時間 |
|--------|---------|
| kushinada Stage 1 | ~4h |
| kushinada Stage 2 | ~3h（Stage 2データは少ないため短縮） |
| ContentVec Stage 1 | ~4h |
| ContentVec Stage 2 | ~3h |
| **合計** | **~14h** |

#### 出力ファイル

```
assets/pretrained_v2/
  jpn_singing_kushinada_G.pth     # kushinada版 Generator
  jpn_singing_kushinada_D.pth     # kushinada版 Discriminator
  jpn_singing_contentvec_G.pth    # ContentVec版 Generator
  jpn_singing_contentvec_D.pth    # ContentVec版 Discriminator
```

**命名規則**: `jpn_singing_{ssl_model}_{G|D}.pth`。既存の `f0G48k.pth` / `f0D48k.pth` と混同しないよう明確に区別する。

---

### 2-12: ターゲット話者ファインチューニング + 品質評価（2日 / GPU ~2h）

#### 概要

作成した事前学習モデル（kushinada版 / ContentVec版）を使って、声優の歌声データ（~10分）でファインチューニングを行い、品質を定量評価する。

#### 評価条件

| 条件 | 事前学習G | 事前学習D | SSLモデル |
|------|----------|----------|----------|
| A (ベースライン) | `f0G48k.pth` (既存) | `f0D48k.pth` (既存) | ContentVec |
| B | `jpn_singing_contentvec_G.pth` | `jpn_singing_contentvec_D.pth` | ContentVec |
| C | `jpn_singing_kushinada_G.pth` | `jpn_singing_kushinada_D.pth` | kushinada |

#### FTパラメータ

```bash
# WebUIから実行する場合（infer-web.py経由）
# または直接:
uv run python infer/modules/train/train.py \
  -se 5 \
  -te 50 \
  -pg assets/pretrained_v2/jpn_singing_kushinada_G.pth \
  -pd assets/pretrained_v2/jpn_singing_kushinada_D.pth \
  -g 0 \
  -bs 4 \
  -e target_singer_kushinada \
  -sr 48k \
  -sw 1 \
  -v v2 \
  -f0 1 \
  -l 0 \
  -c 1
```

FTは10分データなので `-c 1`（GPUキャッシュ）を有効にして高速化。

#### 評価手順（M0の評価基盤を使用）

```bash
# 各条件の変換音声を生成
uv run python tools/infer_cli.py \
  --model_name target_singer_kushinada.pth \
  --input_path test_songs/test1.wav \
  --opt_path results/condition_C/test1.wav \
  --index_path logs/target_singer_kushinada/added_IVF*.index \
  --f0method rmvpe

# M0の評価スクリプトで定量比較
uv run python tools/eval/run_eval.py \
  --ref test_songs/test1_reference.wav \
  --conv results/condition_C/test1.wav
```

#### 評価指標とGo/No-Go基準

| 指標 | ベースライン(条件A)目標 | 日本語事前学習(条件C)目標 | Go条件 |
|------|----------------------|------------------------|--------|
| Whisper CER | 15-25% | 8-15% | **5pt以上改善** |
| MCD | 7.5-8.5 dB | 6.0-7.0 dB | 改善傾向 |
| F0 RMSE | 25-35 cents | 12-20 cents | 改善傾向 |
| FT収束エポック | 50-100 | 3-5 | **大幅短縮** |
| 主観評価 | ベースライン | ベースライン以上 | 劣化なし |

#### FT収束の判定方法

TensorBoardの `loss/g/mel` を監視し、以下の基準でエポック数を記録:

1. `loss/g/mel` が最小値の110%以内に到達したエポック
2. 以降5エポック連続で改善しない場合に収束と判定

---

## 3. エージェントチーム構成

### 推奨チーム構成

| 役割 | 担当タスク | スキル要件 |
|------|-----------|-----------|
| **データエンジニア** | 2-7, 2-7b, 2-8 | Python, librosa, 音声処理の基礎。データセットのライセンス確認能力 |
| **MLエンジニア** | 2-9, 2-10, 2-11 | PyTorch分散学習、VRAM管理、TensorBoard監視。`train.py` の学習ループ理解 |
| **評価担当** | 2-12 | M0評価スクリプトの操作、定量・主観評価の実施。音声の聴覚評価経験 |

### タスク依存関係と並列化

```
2-7 (データセット準備)
  |
  +--> 2-7b (データ拡張)    ←--- 並列可能 --→  2-8 (filelist生成スクリプト)
  |       |                                      |
  +-------+--------------------------------------+
  |
  v
2-9 (Stage 1 事前学習)    ... GPU: ~4h 待ち
  |
  v
2-10 (Stage 2 事前学習)   ... GPU: ~10h 待ち
  |
  v
2-11 (2バリアント作成)    ... GPU: ~14h 待ち
  |
  v
2-12 (FT + 品質評価)     ... GPU: ~2h
```

**並列化のポイント**:
- 2-7bと2-8は独立しており並列実行可能。データエンジニアが2-7b、MLエンジニアが2-8を担当
- GPU学習ジョブ（2-9, 2-10, 2-11）は逐次実行が必要だが、GPU待ち時間に2-12の評価スクリプト準備やテストデータ選定が可能
- 2-11の2バリアント（kushinada版 / ContentVec版）のStage 1は並列実行可能（2GPU環境がある場合）

---

## 4. 提供範囲・テスト項目

### ユニットテスト

| テスト | 対象ファイル | テスト内容 |
|--------|------------|-----------|
| `test_pitch_shift.py` | `tools/augment/pitch_shift.py` | ピッチシフト後のSRが維持されること、ファイル数が期待倍率になること、NaNが含まれないこと |
| `test_time_stretch.py` | `tools/augment/pitch_shift.py` | ストレッチ後の音声長が期待範囲内であること（rate=0.9なら約111%の長さ） |
| `test_gen_filelist.py` | `tools/pretrain/gen_filelist.py` | 生成されるfilelistが正しいフォーマットであること、全カラムのパスが存在すること、speaker_idが連番であること |
| `test_config_pretrain.py` | `configs/v2/48k_pretrain.json` | JSONが正しくパースされること、必須キーがすべて存在すること、`spk_embed_dim`と`mel_fmax`が期待値であること |
| `test_spk_embed_dim.py` | `infer/lib/infer_pack/models.py` | `SynthesizerTrnMs768NSFsid` が `spk_embed_dim=256` で初期化可能であること |
| `test_pretrain_load_strict.py` | `infer/modules/train/train.py` | spk_embed_dimが異なるpretrainモデルのロードで `strict=False` 修正後にエラーが発生しないこと |

### 統合テスト

| テスト | 内容 | 判定基準 |
|--------|------|---------|
| 前処理 -> 特徴量抽出 | JVS-MuSiC 1話者分で前処理 -> F0抽出 -> SSL特徴量抽出を実行 | エラーなく完了、`0_gt_wavs/`, `1_16k_wavs/`, `2_f0/`, `2_f0nsf/`, `3_feature768/` が生成される |
| filelist生成 -> 学習開始 | 生成したfilelistでtrain.pyが起動し、1エポック完走 | `loss/g/total` がNaN/Infにならない |
| データ拡張 -> 前処理 | 拡張済みデータに対してpreprocess.pyが正常に動作 | セグメント化・16kHzリサンプルが正常完了 |
| Stage 1 -> Stage 2 | Stage 1のチェックポイントをStage 2で正しくロードできる | `loaded pretrained` ログが出力される、学習が継続する |
| 事前学習 -> FT | 事前学習モデル(spk_embed_dim=256)を `-pg` / `-pd` に指定してFT(spk_embed_dim=1)が正常に開始 | `loaded pretrained` ログ出力、`emb_g.weight` の形状不一致が自動フォールバックされること |
| process_ckpt互換性 | FTで生成されたチェックポイントをsavee()で推論用に変換 | 推論モデルのconfig内spk_embed_dimがFT時の値(1)で保存されること |

### E2Eテスト

| テスト | 手順 | 期待結果 |
|--------|------|---------|
| 音声変換品質 | 事前学習モデルでFT -> 推論 -> M0評価スクリプト | CER/MCD/F0RMSEの数値が取得できる |
| WebUI互換性 | 事前学習モデルを `assets/pretrained_v2/` に配置し、WebUIから新規学習 | WebUI上でG/Dの事前学習パスを指定してFTが正常実行される |
| 推論互換性 | FTで生成された `.pth` ファイルを `assets/weights/` に配置し推論 | `infer-web.py` / `infer_cli.py` で音声変換が実行できる |

---

## 5. 懸念事項とレビュー項目

### 技術的懸念

| # | 懸念 | 影響度 | 発生確率 | 対策 |
|---|------|--------|---------|------|
| 1 | **JVS-MuSiC 24->48kHz高域欠損** | 高 | 確実 | Stage 1で `mel_fmax=12000` に制限。Stage 2の48kHzネイティブデータで全帯域を再学習。代替案として32kHz configでStage 1を実行 |
| 2 | **spk_embed_dim変更時のweight互換性** | 高 | 確実 | 事前学習(spk_embed_dim=256)とFT(spk_embed_dim=1 or 109)でspk_embed_dimが異なるため、`emb_g.weight` の形状不一致が発生。現行の `train.py` L215-227 の事前学習ロードでは `load_state_dict()` に `strict=False` が**ない**ためRuntimeErrorになる。**コード修正が必要**。修正方法は3案: (a) `net_g.load_state_dict(..., strict=False)` に変更、(b) `emb_g` キーを除外するフィルタリング追加、(c) `utils.py` L99-127 の `load_checkpoint()` 関数（shape不一致時に自動フォールバック + `strict=False`）を事前学習ロードにも適用するリファクタリング。なお `process_ckpt.py` L36 で `hps.model.spk_embed_dim` が推論用モデルのconfigに保存されるため、FT時のconfigで正しいspk_embed_dim(=1)を設定する必要がある |
| 3 | **Stage 1->Stage 2でのmel_fmax変更の影響** | 中 | 中 | Stage 1で12kHz制限のmelを学習した後、Stage 2で全帯域に切り替えると、12kHz以上の領域で急激なloss増加が起きうる。Stage 2初期の学習率を十分低くし（5e-5）、warmup_epochs=3で緩やかに適応させる |
| 4 | **データ拡張によるアーティファクト** | 中 | 低 | +-4半音のピッチシフトはフォルマント構造に影響を与える可能性がある。拡張データのみで事前学習せず、必ず原音データと混合する。拡張データの比率は原音:拡張 = 1:4 程度に制限 |
| 5 | **VRAM不足** | 中 | 低 | RTX 4090 24GBで batch_size=8 は十分だが、データ拡張で学習データ量が7倍になるとエポックあたりのステップ数が増加しGPU時間が長くなる。batch_size=6に下げるか、エポック数を減らすことで対応 |
| 6 | **Windows環境での分散学習** | 中 | 高 | `train.py` L120 で `backend="gloo"` を使用しており、NCCL（Linux専用）は不使用。単一GPU学習なら問題ないが、マルチGPU時はWSL2が推奨 |

### レビュー項目

| # | レビュー観点 | 確認内容 |
|---|------------|---------|
| 1 | **ライセンスコンプライアンス** | 各データセットのライセンスに従った利用であること。きりたん・イタコの「研究のみ」制約、GTSingerの「研究OS」制約を遵守。JVS-MuSiCの再配布禁止を遵守（データセットをgitに含めない） |
| 2 | **既存コードへの影響** | 新規作成ファイル（`tools/augment/`, `tools/pretrain/`, `configs/v2/48k_pretrain.json`）は既存コードに影響を与えないこと。`train.py` の修正（spk_embed_dim互換性）が既存FTワークフローを壊さないこと |
| 3 | **チェックポイント命名規則** | 事前学習モデルのファイル名が既存の `f0G48k.pth` / `f0D48k.pth` と衝突しないこと |
| 4 | **GPU時間の妥当性** | 合計GPU時間（~30h）がRTX 4090で2日以内に完了可能であること。Cloud GPU利用の場合のコスト見積もり（$150-250） |
| 5 | **process_ckpt.py との互換性** | `infer/lib/train/process_ckpt.py` の `savee()` 関数（L12-47）は `hps.model.spk_embed_dim` を `opt["config"]` に保存する。spk_embed_dim=256の事前学習モデルを使ってFTした場合、最終的な推論用モデルのspk_embed_dimが正しく1になることを確認 |

---

## 6. 一から作り直すとしたら（M2フェーズ全体の理想設計）

### 現行設計の制約

現行のRVC学習パイプラインは以下の前提で設計されている:

1. **単一話者FT前提**: `spk_embed_dim=109` は元の事前学習データの話者数にハードコードされている。多話者事前学習を想定した柔軟なspk_embed_dim管理機構がない
2. **fairseq依存**: SSL特徴量抽出が `fairseq.checkpoint_utils.load_model_ensemble_and_task()` に依存しており（`extract_feature_print.py` L87-90）、HuggingFace transformersモデルの利用が困難
3. **filelist形式の固定**: `|` 区切り5カラム（audio, phone, pitch, pitchf, spk_id）が `data_utils.py` にハードコードされており、メタデータの拡張（データセット名、拡張種別等）が困難
4. **configの分散管理**: `48k.json` の値がCLI引数（`train.py` の `get_hparams()`）で上書きされるため、実際の学習パラメータの追跡が難しい

### 理想的なM2設計

**1. SSLモデル抽象化レイヤー**

```python
# infer/modules/ssl/base.py
class SSLExtractor(ABC):
    @abstractmethod
    def extract_features(self, wav_16k: Tensor) -> Tensor: ...
    @property
    def output_dim(self) -> int: ...
    @property
    def output_layer(self) -> int: ...

# ContentVec, kushinada, rinna, Spin V2 がこのインタフェースを実装
```

M2-Aで実装予定のタスク2-1, 2-2がこれに相当する。

**2. 柔軟なspk_embed_dim管理**

```python
# models.py の SynthesizerTrnMs768NSFsid.__init__() で
# spk_embed_dim を動的に変更可能にする
class SynthesizerTrnMs768NSFsid(nn.Module):
    def resize_speaker_embedding(self, new_spk_embed_dim):
        """事前学習 -> FT間のspk_embed_dim変更を安全に実行"""
        old_weight = self.emb_g.weight.data
        self.emb_g = nn.Embedding(new_spk_embed_dim, self.gin_channels)
        # 既存話者のweightをコピー
        n_copy = min(old_weight.size(0), new_spk_embed_dim)
        self.emb_g.weight.data[:n_copy] = old_weight[:n_copy]
```

**3. 段階的事前学習の自動化**

```python
# tools/pretrain/run_pretrain_pipeline.py
# Stage 1 -> Stage 2 -> バリアント作成 をCLI一発で実行
python tools/pretrain/run_pretrain_pipeline.py \
  --stage1_data datasets/pretrain/jvs_music \
  --stage2_data datasets/pretrain/nit_song070 datasets/pretrain/kiritan ... \
  --ssl_models contentvec kushinada \
  --output_dir assets/pretrained_v2/jpn_singing/ \
  --gpu 0 \
  --config configs/v2/48k_pretrain.json
```

**4. Experiment管理**

現行は `logs/{experiment_name}/` にフラットに保存されるが、理想的にはmlflowやWandBによる実験管理を導入し、ハイパーパラメータ・メトリクス・チェックポイントを統一管理する。ただし、これはM2の範囲では過剰投資であり、M4以降の検討事項とする。

### 現実的な妥協点

M2-Bでは上記の理想設計を完全に実装することはスコープ外とし、以下の最小限の変更で対応する:

1. **spk_embed_dim**: 事前学習用config（`48k_pretrain.json`）を別ファイルとして管理。FT用configは既存の `48k.json`（spk_embed_dim=109）を維持
2. **SSL特徴量**: M2-Aの成果（SSLローダー抽象化）を利用。fairseq/transformersの切替はM2-Aで解決済みの前提
3. **filelist**: 既存フォーマットを維持しつつ、`gen_filelist.py` でspk_idの自動採番を行う
4. **実験管理**: TensorBoardとgitタグで最低限の追跡を行う

---

## 7. 後続タスクへの連絡事項

### M3-A（損失関数改善 Week 6）への引き継ぎ

1. **事前学習モデルの命名規則**: `assets/pretrained_v2/jpn_singing_{ssl_model}_{G|D}.pth` で統一。M3でSnakeBeta導入後に再学習する際も同じ規則を適用（`jpn_singing_kushinada_snakebeta_G.pth` 等）
2. **mel_fmax設定**: M2-B Stage 1では `mel_fmax=12000` に制限した。M3-Bでmel_fmin=40への変更と同時に、mel_fmaxもnull（全帯域）に戻すことを推奨。SnakeBeta事前学習の再実行時にまとめて適用する
3. **spk_embed_dim**: M2-Bの事前学習では256を使用。M3での再学習時も256を維持し、FT時のみ1に変更するパターンを踏襲する
4. **train.py のstrict=False修正**: M2-Bで `train.py` L215-227 の事前学習ロードに `strict=False` を追加（または `utils.py` の `load_checkpoint()` を再利用するリファクタリング）を実施済みの前提。M3-Bの事前学習再実行時もこの修正が必要

### M3-B（ボコーダ改善 Week 7-8）への引き継ぎ

1. **事前学習の再実行が必要**: M3-BでSnakeBeta活性化関数やアンチエイリアスフィルタを導入する場合、モデルアーキテクチャが変わるため、M2-Bの事前学習モデルは**使用不可**。M3-Bで改めてStage 1 -> Stage 2の事前学習を再実行する必要がある
2. **データパイプラインの再利用**: M2-Bで構築したデータ拡張スクリプト（`tools/augment/`）、filelist生成スクリプト（`tools/pretrain/`）、前処理済みデータはそのまま再利用可能。SSL特徴量（`3_feature768/`）も再利用可能（ボコーダ変更はSSL特徴量に影響しない）
3. **GPU時間の見積もり**: M3-Bの事前学習再実行はM2-Bと同等のGPU時間（~14-30h）が必要。milestones.md のM3-B GPU見積もりに反映済み

### ターゲット話者FTを行うエンドユーザーへの影響

1. **WebUI変更**: M2-AのタスクM2-6bで「SSLモデル選択UI」を実装済みの前提。事前学習モデル選択ドロップダウンに `jpn_singing_kushinada` / `jpn_singing_contentvec` の選択肢を追加する必要がある
2. **FTパラメータの推奨値**: 日本語歌声事前学習モデルを使用する場合のFT推奨エポック数は3-10（現行のデフォルト20000から大幅削減）。WebUIのデフォルト値またはプリセットでの反映を検討
3. **ドキュメント**: SSLモデル選択ガイド（milestones.md「ドキュメント計画」M2）に、事前学習モデルの選択指針を追加。「歌声変換 -> jpn_singing_kushinada推奨」「話し声変換 -> 既存pretrained_v2推奨」

### Go/No-Go判定後のフォールバック

Go/No-Go判定でNo-Goとなった場合の対応:

| No-Go理由 | フォールバック策 | 所要時間 |
|-----------|----------------|---------|
| kushinadaがCER未改善 | rinna/japanese-hubert-baseに切り替え。BOOTHのnadare氏チェックポイント(`f0X48k768_jphubert_v2`)を利用可能 | 0.5日 |
| 事前学習効果不足 | Stage 2データにNo.7歌唱DB、JSUT-songを追加。エポック数増加（300->500） | 1日 + GPU 5h |
| 歌声品質劣化 | データ拡張比率の再調整（原音比率を増やす）。mel_fmax制限値の見直し | 0.5日 |
| VRAM不足 | Cloud GPU（A100/H100）に移行。コスト$150-250 | 即時 |
