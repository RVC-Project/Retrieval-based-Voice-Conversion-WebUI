# M0: 評価基盤構築

## メタ情報
- **マイルストーン**: M0
- **フェーズ**: Week 1（M1と並列実行）
- **工数見積もり**: 3人日
- **GPU要件**: RTX 4070 Ti Super 16GB（Whisper推論用）
- **前提タスク**: なし
- **ステータス**: 未着手
- **関連マイルストーン**: [milestones.md](../milestones.md) > M0セクション

---

## 1. タスク目的とゴール

### 目的

日本語歌声変換プロジェクト全体を通じて、モデル改善の効果を定量的に追跡するための評価基盤を構築する。現状のRVC WebUIには音声変換品質を客観的に測定する仕組みが存在せず、改善の効果検証が主観評価に依存している。M1以降のすべてのマイルストーンで「改善が測定可能であること」を保証するため、M0で評価パイプラインを先行して整備する。

### なぜ必要か

1. **再現性**: 開発者間で「良くなった/悪くなった」の判断基準がブレるのを防ぐ。JSON出力による定量値で議論できるようにする
2. **Go/No-Go判定の根拠**: M1終了時のGo/No-Go判定（MCD 5%以上改善、F0 RMSE 10%以上改善）には、M0で取得したベースライン数値が必須
3. **回帰検出**: パラメータ変更やモデル差し替えが既存品質を劣化させていないことを即座に検出する
4. **M1との並列実行**: M1の設定変更（Dropout追加、segment_size拡張等）は評価スクリプト完成前でも安全に適用できるが、M1-11「評価実行+ベースライン比較」にはM0の成果物が必要

### 成功条件

1. `uv run python tools/eval/run_eval.py --ref ref.wav --conv conv.wav` を実行すると、MCD / F0 RMSE / Whisper CER / PASS/WARN/FAIL判定を含むJSON がstdoutに出力される
2. 現行ContentVecモデル（変更前）のベースライン値が `tools/eval/baselines/` に記録されている
3. 各指標の計算結果が学術論文の手法と一致することをユニットテストで検証済み
4. Whisper CERについて、原音（参照音声そのもの）のCERがベースラインとして記録されている（歌声Whisper信頼性の検証用）

---

## 2. 実装する内容の詳細

### サブタスク一覧

- [ ] **0-1: 評価スクリプト基盤作成**（1日）
  - CLI エントリーポイント `tools/eval/run_eval.py` の実装
  - argparse によるインターフェース設計
  - 各メトリクスモジュールの呼び出しとJSON集約
  - PASS/WARN/FAIL 判定ロジック
  - ログ出力（`--verbose` オプション）
  
- [ ] **0-2: MCD + F0 RMSE 自動計測**（0.5日）
  - `tools/eval/metrics/mcd.py` - Mel Cepstral Distortion計算
  - `tools/eval/metrics/f0_accuracy.py` - F0 RMSE計算（cent単位）
  - DTWによる時間軸アラインメント
  - 無声区間のハンドリング

- [ ] **0-3: Whisper CERパイプライン**（1日）
  - `tools/eval/metrics/whisper_cer.py` - Whisper転写 + CER計算
  - 日本語テキスト正規化（全角/半角、カタカナ/ひらがな統一等）
  - 歌声向けWhisperパラメータ調整
  - 原音CERの同時計測機能

- [ ] **0-4: ベースライン測定**（0.5日）
  - 現行ContentVecモデルで変換した音声サンプルの各指標値を記録
  - 結果を `tools/eval/baselines/` にJSON保存
  - 原音のWhisper CERも同時記録（Whisper信頼性のベースライン）

### 変更対象ファイル・新規作成ファイル

#### 新規作成

```
tools/eval/
  __init__.py              # パッケージ初期化
  run_eval.py              # CLI エントリーポイント（メイン）
  metrics/
    __init__.py            # メトリクスパッケージ初期化
    mcd.py                 # Mel Cepstral Distortion
    f0_accuracy.py         # F0 RMSE（cent単位）
    whisper_cer.py         # Whisper CER（日本語正規化込み）
  baselines/
    README.md              # ベースライン記録の説明（任意）
    baseline_contentvc.json # ContentVecモデルのベースライン値
```

#### 変更対象（既存ファイル）

| ファイル | 変更内容 |
|---------|---------|
| `pyproject.toml` | `openai-whisper`, `jiwer`, `jaconv`, `fastdtw` を dependencies に追加 |

#### 参照のみ（変更なし）

| ファイル | 参照理由 |
|---------|---------|
| `infer/modules/vc/pipeline.py` | F0抽出ロジック（RMVPE）の仕様確認。特にF0範囲 50-1100Hz、16kHz固定の仕様 |
| `infer/modules/vc/modules.py` | VCクラスの推論フロー確認。`vc_single()` の戻り値形式 |
| `infer/lib/rmvpe.py` | RMVPE F0抽出の内部実装。評価スクリプトでのF0抽出に同じアルゴリズムを使用するか検討 |
| `configs/config.py` | デバイス検出ロジック。評価スクリプトでも同じ `Config` クラスを利用可能 |
| `configs/v2/48k.json` | モデル設定（`sampling_rate: 48000`, `hop_length: 480`, `n_mel_channels: 128`, `mel_fmin: 0.0`）。MCD計算のmel抽出パラメータに使用 |
| `tools/infer_cli.py` | 既存CLIの引数設計パターンを踏襲 |

### 技術仕様

#### 2.1 CLI インターフェース (`run_eval.py`)

```
usage: run_eval.py [-h] --ref REF --conv CONV
                   [--metrics {all,mcd,f0,cer}]
                   [--whisper-model {tiny,base,small,medium,large-v3}]
                   [--whisper-lang ja]
                   [--ref-text REF_TEXT]
                   [--config CONFIG]
                   [--device {cuda,cpu}]
                   [--output OUTPUT]
                   [--verbose]

引数:
  --ref REF             参照音声（原音）のパス（WAV形式）
  --conv CONV           変換音声のパス（WAV形式）
  --metrics METRICS     計測する指標（カンマ区切り）。デフォルト: all
  --whisper-model MODEL Whisperモデルサイズ。デフォルト: medium
  --whisper-lang LANG   Whisper言語指定。デフォルト: ja
  --ref-text TEXT       参照テキスト（CER計算用）。省略時はWhisperで参照音声を転写
  --config CONFIG       モデル設定ファイルのパス（MCD等のmelパラメータ取得用）。デフォルト: configs/v2/48k.json
  --device DEVICE       推論デバイス。デフォルト: 自動検出
  --output OUTPUT       JSON出力先ファイルパス。省略時はstdout
  --verbose             詳細ログ出力
```

#### 2.2 JSON出力スキーマ

```json
{
  "version": "0.1.0",
  "timestamp": "2026-04-11T12:00:00+09:00",
  "files": {
    "reference": "path/to/ref.wav",
    "converted": "path/to/conv.wav"
  },
  "metrics": {
    "mcd": {
      "value": 7.82,
      "unit": "dB",
      "status": "WARN",
      "thresholds": {"PASS": "<6.0", "WARN": "6.0-8.0", "FAIL": ">8.0"}
    },
    "f0_rmse": {
      "value": 28.5,
      "unit": "cents",
      "status": "WARN",
      "thresholds": {"PASS": "<20", "WARN": "20-50", "FAIL": ">50"},
      "details": {
        "voiced_frame_ratio": 0.72,
        "vuv_error_rate": 0.05
      }
    },
    "whisper_cer": {
      "value": 0.182,
      "unit": "ratio",
      "status": "WARN",
      "thresholds": {"PASS": "<0.10", "WARN": "0.10-0.20", "FAIL": ">0.20"},
      "details": {
        "ref_text": "きみのこえがきこえる",
        "conv_text": "きみのこえがきえる",
        "ref_source": "whisper",
        "ref_audio_cer": 0.03
      }
    }
  },
  "overall_status": "WARN",
  "config": {
    "whisper_model": "medium",
    "whisper_lang": "ja",
    "f0_method": "rmvpe",
    "sample_rate": 48000,
    "config_file": "configs/v2/48k.json"
  }
}
```

`overall_status` は全指標の中で最も悪い判定を採用する（FAIL > WARN > PASS）。

#### 2.3 MCD 計算仕様 (`metrics/mcd.py`)

**アルゴリズム**: Mel Cepstral Distortion（MCD-DTW）

```python
def compute_mcd(ref_path: str, conv_path: str, sr: int = 48000, n_mels: int = 128,
                n_mfcc: int = 13, fmin: float = 0.0, fmax: float | None = None,
                hop_length: int = 480) -> dict:
    """
    MCD-DTW を計算する。

    1. ref/conv をロードし、同一サンプリングレートにリサンプル
    2. MFCC抽出（0次係数を除く1-12次元を使用）
    3. fastdtw で時間軸アラインメント
    4. MCD = (10 * sqrt(2) / ln(10)) * mean(||mfcc_ref - mfcc_conv||_2)

    Returns:
        {"value": float, "unit": "dB", "frames_aligned": int}
    """
```

- **MFCC抽出パラメータ**: 既存の `configs/v2/48k.json` と整合させる（`sr=48000`, `hop_length=480`, `n_mel_channels=128`, `mel_fmin=0.0`）
- **MCD公式**: `MCD [dB] = (10 * sqrt(2) / ln(10)) * mean(sqrt(sum((mfcc_ref_i - mfcc_conv_i)^2)))`
- **MFCC次元**: 0次（パワー）を除く1-12次元。13次以降はノイズが多く不安定
- **DTWアラインメント**: `fastdtw` ライブラリ（radius=1 でコスト削減）。歌声は原音と変換音声でテンポが一致するためradiusを小さくできる
- **リサンプル**: 参照と変換の `sr` が異なる場合、`librosa.resample` で統一

#### 2.4 F0 RMSE 計算仕様 (`metrics/f0_accuracy.py`)

**アルゴリズム**: F0 RMSE（cent単位） + Voiced/Unvoiced Error Rate

```python
def compute_f0_rmse(ref_path: str, conv_path: str, sr: int = 48000,
                     hop_length: int = 480, f0_method: str = "rmvpe",
                     f0_min: float = 50, f0_max: float = 1100) -> dict:
    """
    F0 RMSE を cent 単位で計算する。

    1. ref/conv からF0抽出（RMVPEを使用）
    2. 両方が有声のフレームのみ抽出
    3. fastdtw で時間軸アラインメント
    4. cent変換: 1200 * log2(f0_conv / f0_ref)
    5. RMSE = sqrt(mean(cent_diff^2))
    6. VUV Error Rate = 有声/無声の不一致率

    Returns:
        {"value": float, "unit": "cents", "voiced_frame_ratio": float,
         "vuv_error_rate": float, "frames_total": int}
    """
```

- **F0抽出**: 既存の `infer/lib/rmvpe.py` の `RMVPE` クラスを再利用する。`pipeline.py` と同じ `rmvpe.pt` モデルを使用
- **cent変換**: `cent = 1200 * log2(f0_conv / f0_ref)`。人間の音高知覚に対応する対数スケール
- **無声区間の扱い**: F0=0 のフレームは無声と判定。RMSE計算では両方が有声のフレームのみを使用
- **VUV Error Rate**: `(有声を無声と誤判定 + 無声を有声と誤判定) / 全フレーム数`
- **DTWアラインメント**: MCD と同じ `fastdtw` を使用。F0シーケンスに対してアラインメントを実行

#### 2.5 Whisper CER 計算仕様 (`metrics/whisper_cer.py`)

**アルゴリズム**: Whisper転写 + jiwer CER + 日本語正規化

```python
def compute_whisper_cer(ref_path: str, conv_path: str,
                        ref_text: str | None = None,
                        model_name: str = "medium",
                        language: str = "ja",
                        device: str = "cuda") -> dict:
    """
    Whisper CER を計算する。

    1. ref_text が未指定の場合、参照音声をWhisperで転写（ref_audio_cer も計算）
    2. 変換音声をWhisperで転写
    3. 日本語テキスト正規化
       - 全角英数字 → 半角
       - カタカナ → ひらがな（jaconv）
       - 句読点・記号の除去
       - 連続空白の正規化
    4. jiwer.cer() で CER 計算

    Returns:
        {"value": float, "unit": "ratio", "ref_text": str, "conv_text": str,
         "ref_source": "provided" | "whisper", "ref_audio_cer": float | None}
    """
```

- **Whisperモデル**: デフォルトは `medium`（769M params）。RTX 4070 Ti Super 16GBで十分動作する。`large-v3` も選択可能だが推論時間が長い
- **言語指定**: `language="ja"` を明示的に指定し、言語検出のオーバーヘッドと誤検出を回避
- **日本語正規化パイプライン**:
  1. `jaconv.z2h(text, kana=False, digit=True, ascii=True)` - 全角英数字を半角に
  2. `jaconv.kata2hira(text)` - カタカナをひらがなに統一
  3. 正規表現で句読点・記号を除去: `re.sub(r'[、。！？\s\.,!?]', '', text)`
  4. NFKCユニコード正規化: `unicodedata.normalize('NFKC', text)`
- **原音CER**: 参照音声自体をWhisperで転写したときのCER。歌声の場合は原音でもCERが高くなる（メリスマ・母音引き伸ばし等）ため、変換音声のCERと比較する際のベースラインとして使用
- **Whisper歌声向けパラメータ**:
  - `beam_size=5`（デフォルトより大きめ）
  - `best_of=5`
  - `temperature=0`（確定的デコード）
  - `condition_on_previous_text=False`（歌詞の繰り返しによるループ防止）
  - `no_speech_threshold=0.3`（間奏部分の誤転写防止）
  - `compression_ratio_threshold=2.8`（歌声は圧縮比が高くなりやすいため閾値を緩和）

#### 2.6 PASS/WARN/FAIL 閾値

| 指標 | PASS | WARN | FAIL |
|------|------|------|------|
| Whisper CER | <10% (0.10) | 10-20% (0.10-0.20) | >20% (0.20) |
| MCD | <6.0 dB | 6.0-8.0 dB | >8.0 dB |
| F0 RMSE | <20 cents | 20-50 cents | >50 cents |

閾値は `run_eval.py` 内に定数として定義する。将来的にYAML設定ファイルで外部化可能な設計にしておく。

```python
THRESHOLDS = {
    "mcd": {"pass": 6.0, "fail": 8.0, "unit": "dB", "lower_is_better": True},
    "f0_rmse": {"pass": 20.0, "fail": 50.0, "unit": "cents", "lower_is_better": True},
    "whisper_cer": {"pass": 0.10, "fail": 0.20, "unit": "ratio", "lower_is_better": True},
}
```

#### 2.7 依存パッケージ

`pyproject.toml` に追加するパッケージ:

| パッケージ | バージョン | 用途 |
|-----------|----------|------|
| `openai-whisper` | `>=20240930` | 音声転写（CER計算） |
| `jiwer` | `>=3.0.0` | CER/WER計算 |
| `jaconv` | `>=0.3.4` | 日本語テキスト正規化（全角半角、カタカナひらがな変換） |
| `fastdtw` | `>=0.3.4` | DTWアラインメント（MCD、F0のフレーム対応付け） |

既存パッケージで使用するもの（追加不要）:

| パッケージ | 用途 |
|-----------|------|
| `librosa` | MFCC抽出、リサンプル |
| `numpy` | 数値計算全般 |
| `scipy` | 統計計算 |
| `torch` | RMVPE F0抽出、Whisper推論 |
| `soundfile` | 音声ファイル読み込み |

#### 2.8 ベースライン記録フォーマット

`tools/eval/baselines/baseline_contentvc.json`:

```json
{
  "model": "ContentVec (v2/48k, 現行デフォルト)",
  "date": "2026-04-XX",
  "hardware": "RTX 4070 Ti Super 16GB",
  "test_samples": [
    {
      "name": "sample_01",
      "ref": "path/to/ref_01.wav",
      "conv": "path/to/conv_01.wav",
      "description": "J-POP女声、中音域",
      "metrics": {
        "mcd": {"value": 8.1, "status": "FAIL"},
        "f0_rmse": {"value": 30.2, "status": "WARN"},
        "whisper_cer": {"value": 0.19, "status": "WARN"},
        "whisper_cer_ref_audio": {"value": 0.04, "note": "原音のCER"}
      }
    }
  ],
  "summary": {
    "mcd_mean": 8.1,
    "f0_rmse_mean": 30.2,
    "whisper_cer_mean": 0.19,
    "whisper_cer_ref_audio_mean": 0.04,
    "overall_status": "FAIL"
  }
}
```

テストサンプルは最低3曲分を用意する（音域・テンポ・ジャンルの多様性を確保）。

---

## 3. エージェントチーム構成

### 推奨構成: 3エージェント

| 役割 | 担当範囲 | 工数 |
|------|---------|------|
| **実装エージェント A** | `run_eval.py` (0-1) + `mcd.py` + `f0_accuracy.py` (0-2) | 1.5日 |
| **実装エージェント B** | `whisper_cer.py` (0-3) + 日本語正規化ロジック | 1日 |
| **テスト・計測エージェント** | ユニットテスト作成 + ベースライン測定 (0-4) + `pyproject.toml` 更新 | 0.5日 |

### 実行順序

```
Day 1:
  エージェントA → run_eval.py 基盤 + mcd.py + f0_accuracy.py
  エージェントB → whisper_cer.py（並列作業可）

Day 2:
  エージェントA → run_eval.py に全メトリクス統合 + PASS/WARN/FAIL判定
  エージェントB → 日本語正規化の追い込み + エッジケース対応

Day 3:
  テスト・計測エージェント → ユニットテスト + pyproject.toml更新 + ベースライン測定
  全員 → 統合テスト + レビュー
```

### エージェント間のインターフェース合意事項

各メトリクスモジュールは以下の統一インターフェースに従う:

```python
# 各 metrics/*.py が公開する関数の共通シグネチャ
def compute_<metric_name>(ref_path: str, conv_path: str, **kwargs) -> dict:
    """
    Returns:
        {
            "value": float,          # メトリクスの数値
            "unit": str,             # 単位 ("dB", "cents", "ratio")
            "details": dict | None,  # 追加情報（オプション）
        }
    """
```

`run_eval.py` は各モジュールの `compute_*` 関数をインポートし、閾値判定・JSON集約を担当する。

---

## 4. 提供範囲・テスト項目

### スコープ

#### In Scope

- MCD (Mel Cepstral Distortion) の自動計測
- F0 RMSE (cent単位) の自動計測
- Whisper CER (日本語正規化込み) の自動計測
- PASS/WARN/FAIL 判定ロジック
- JSON出力（stdout + ファイル出力）
- 現行ContentVecモデルのベースライン値の記録
- 原音のWhisper CERベースライン記録
- `pyproject.toml` への依存パッケージ追加

#### Out of Scope

- **話者類似度 (Speaker Similarity)**: M0では実装しない。話者埋め込みモデル（WeSpeaker/ECAPA-TDNN等）の選定が必要であり、M1以降のタスクとして別チケット化する
- **フォルマント分析**: 歌声品質の補助指標として有用だが、優先度が低いためM0スコープ外
- **ビブラート分析**: DWT基盤のビブラート保存評価はM3のタスク3-5で実装予定
- **主観MOS**: 自動化が困難。開発者3名による簡易MOSはM1-11で手動実施
- **バッチ評価**: 複数ファイルの一括評価機能。M0ではファイルペア単位の評価のみ
- **WebUI統合**: 評価機能のGradio UIへの組み込み
- **CI/CD統合**: GitHub Actionsでの自動評価パイプライン

### ユニットテスト

テストファイル: `tests/eval/` ディレクトリに配置する。

#### MCD テスト (`tests/eval/test_mcd.py`)

| テストケース | 内容 | 期待結果 |
|------------|------|---------|
| `test_mcd_identical` | 同一WAVファイルのMCD | MCD = 0.0 dB |
| `test_mcd_known_value` | 事前計算済みのペアでMCD検証 | 既知の値 +/- 0.5 dB |
| `test_mcd_different_sr` | 異なるサンプルレートのペア | リサンプル後に正常計算 |
| `test_mcd_short_audio` | 1秒未満の短い音声 | エラーなく計算完了 |
| `test_mcd_mono_stereo` | モノラル vs ステレオ | ステレオをモノラル変換して計算 |

#### F0 RMSE テスト (`tests/eval/test_f0_accuracy.py`)

| テストケース | 内容 | 期待結果 |
|------------|------|---------|
| `test_f0_identical` | 同一WAVのF0 RMSE | F0 RMSE = 0.0 cents |
| `test_f0_pitch_shifted` | 1半音（100 cents）ずらした合成音声 | RMSE ≈ 100 cents |
| `test_f0_silent_audio` | 無音ファイル | vuv_error_rate の適切な値、RMSE は NaN または 0 |
| `test_f0_vuv_error` | 有声/無声の判定確認 | VUV error rate が妥当な範囲 |

#### Whisper CER テスト (`tests/eval/test_whisper_cer.py`)

| テストケース | 内容 | 期待結果 |
|------------|------|---------|
| `test_cer_identical_text` | 同一テキストのCER | CER = 0.0 |
| `test_cer_known_pair` | 既知の参照/仮説テキストペア | jiwer と一致する値 |
| `test_normalize_japanese` | 日本語正規化の個別テスト | カタカナ→ひらがな、全角→半角が正しく変換 |
| `test_normalize_punctuation` | 句読点除去 | 「、」「。」「！」が除去される |
| `test_whisper_with_ref_text` | `--ref-text` 指定時の動作 | 参照テキストが正しく使用される |

#### 統合テスト (`tests/eval/test_run_eval.py`)

| テストケース | 内容 | 期待結果 |
|------------|------|---------|
| `test_full_pipeline` | `run_eval.py` をsubprocessで実行 | JSON出力が正しいスキーマに従う |
| `test_metrics_selection` | `--metrics mcd,f0` で指標選択 | 指定した指標のみ出力 |
| `test_output_file` | `--output result.json` でファイル出力 | ファイルに正しいJSONが書き込まれる |
| `test_pass_warn_fail` | 各ステータスの判定 | 閾値に基づく正しい判定 |
| `test_missing_args` | `--ref` のみ指定、`--conv` なし | わかりやすいエラーメッセージが表示される |
| `test_invalid_file_path` | 存在しないファイルパスを指定 | FileNotFoundError相当のエラーメッセージ |
| `test_non_wav_input` | テキストファイルをWAVとして渡す | 適切なエラーメッセージ（破損ファイル対応） |

### テストデータ

テスト用音声データは以下の方法で準備する:

1. **合成音声**: `scipy.signal` でサイン波を生成し、テスト用WAVとして使用（リポジトリにコミット可能なサイズ）
2. **ピッチシフト音声**: `librosa.effects.pitch_shift` で既知のシフト量を適用した音声ペア
3. **実際の歌声データ**: ベースライン測定時に使用する実データ（gitignoreでリポジトリからは除外し、パスのみ記録）

テストデータ格納先: `tests/eval/fixtures/`（合成音声のみ）

### E2Eテスト

E2Eテストはベースライン測定 (0-4) を兼ねる:

1. 実際の歌声データ（声優ボーカル）を現行モデルで変換
2. `run_eval.py` で全指標を計測
3. JSON出力の全フィールドが正しく埋まっていることを確認
4. ベースライン値が `milestones.md` の「現状推定」範囲内であることを確認:
   - MCD: 7.5-8.5 dB
   - F0 RMSE: 25-35 cents
   - Whisper CER: 15-25%

---

## 5. 懸念事項とレビュー項目

### 実装上の懸念

#### 懸念1: Whisper CER の歌声信頼性（影響度: 高、発生確率: 高）

**問題**: Whisperは話し声で学習されているため、歌声の転写精度が低い。特にメリスマ（音を伸ばしながら音程を変える歌唱技法）、長母音、ファルセットで誤認識が増加する。歌声の原音でも CER 10-20% になる可能性がある。

**対策**:
- 原音のWhisper CERを必ずベースラインとして記録する（`ref_audio_cer` フィールド）
- 変換音声のCERは `conv_cer - ref_audio_cer` の差分で評価する運用ガイドラインを文書化
- `condition_on_previous_text=False` と `compression_ratio_threshold=2.8` で歌声向けにパラメータ調整
- M1完了時にWhisper CERの有効性を再評価し、信頼性が低い場合は開発者3名の簡易MOSを主指標に切り替える

#### 懸念2: DTW アラインメントの精度

**問題**: RVC変換では音声の長さが厳密に保存されるため、DTWは高い精度で動作するはず。ただし、無音区間のトリミング差異があると誤ったアラインメントになる可能性がある。

**対策**:
- DTW前に先頭・末尾の無音をトリムする前処理を追加
- DTWのコスト行列を可視化する `--verbose` オプションで問題を検出可能にする
- radius=1 で計算コストを抑えつつ、問題がある場合はradiusを増やせるようにする

#### 懸念3: MCD の mel パラメータ整合性

**問題**: MCD計算時のmel抽出パラメータ（`n_mels`, `fmin`, `fmax`, `hop_length`）がモデルの学習設定と異なると、数値が不正確になる。

**対策**:
- `configs/v2/48k.json` の値（`n_mel_channels=128`, `mel_fmin=0.0`, `mel_fmax=null`, `hop_length=480`）をデフォルトとして使用
- 将来のconfig変更（M3-Bでのmel_fmin変更等）に追従できるよう、configファイルから動的に読み取るオプションを用意

#### 懸念4: RMVPE モデルの依存

**問題**: F0 RMSE計算で `infer/lib/rmvpe.py` を再利用するが、`assets/rmvpe/rmvpe.pt` モデルファイルが必要。CI環境やモデルファイルがない環境ではF0計算が失敗する。

**対策**:
- RMVPE が見つからない場合は `pyworld.harvest` にフォールバックする
- テストでは合成サイン波を使い、RMVPEなしでも基本テストが通るようにする

#### 懸念5: openai-whisper のバージョン互換性

**問題**: `openai-whisper` は PyTorch バージョンとの互換性がシビアな場合がある。現行は `torch==2.10.0`。また、`openai-whisper` は内部で `tiktoken` 等の依存を持ち、既存の `fairseq` 依存（git版）と競合する可能性がある。

**対策**:
- `uv sync` でインストール後に `import whisper; whisper.load_model("tiny")` が動作することを確認
- 互換性問題が発生した場合は `faster-whisper` (CTranslate2 ベース) への切り替えを検討
- `fairseq` との依存競合が発生した場合は、Whisper推論を別プロセスで実行する設計も検討

#### 懸念6: 32k モデルへの対応

**問題**: 現在 `configs/v2/32k.json` も存在し、`n_mel_channels=80`, `hop_length=320` と48k版とは異なるパラメータを使用している。MCD計算パラメータが48k固定だと、32kモデルの評価で不正確な結果になる。

**対策**:
- デフォルトは48k設定で問題ないが、将来の拡張として `--config` オプションで設定ファイルを指定可能にする（CLI仕様のセクション2.1にも `--config` オプションを追加しておく）
- 当面は48kモデルのみを評価対象とし、32kモデルの評価は必要時に対応する

### レビューチェックリスト

- [ ] **CLI**: `--ref` と `--conv` の両方を指定しないとエラーメッセージが表示されるか
- [ ] **CLI**: 存在しないファイルパスを指定した場合、わかりやすいエラーが出るか
- [ ] **JSON**: 出力スキーマが「2.2 JSON出力スキーマ」に一致しているか
- [ ] **JSON**: `json.loads()` でパースできるか（末尾カンマやコメントがないか）
- [ ] **MCD**: 同一ファイルで MCD=0 になるか
- [ ] **MCD**: `configs/v2/48k.json` のmelパラメータと整合しているか
- [ ] **F0**: cent変換の公式が `1200 * log2(f0_conv / f0_ref)` であることを確認
- [ ] **F0**: F0=0 の無声フレームが正しく除外されているか
- [ ] **CER**: 日本語正規化で「カタカナ→ひらがな」が正しく動作するか
- [ ] **CER**: 全角数字「１２３」が半角「123」に変換されるか
- [ ] **CER**: 原音CER (`ref_audio_cer`) が記録されるか
- [ ] **閾値**: PASS/WARN/FAIL の境界値が正しいか（境界値テスト）
- [ ] **デバイス**: `--device cpu` で全機能が動作するか
- [ ] **pyproject.toml**: 追加パッケージのバージョン指定が適切か
- [ ] **コードスタイル**: `ruff` のlintルール（`pyproject.toml` の `[tool.ruff]` セクション）に準拠しているか
- [ ] **ログ**: `--verbose` で処理の進捗がわかるログが出力されるか
- [ ] **エラーハンドリング**: Whisperモデルのダウンロード失敗時に適切なメッセージが表示されるか
- [ ] **メモリ**: Whisper medium + 5分音声で RTX 4070 Ti Super 16GB のVRAMに収まるか

---

## 6. 一から作り直すとしたら

このフェーズ全体を制約なしで一から設計し直すなら、以下の設計を採用する。

### 6.1 評価フレームワーク化

現在の設計は「1回のCLI呼び出しで1ペアを評価」という最小構成だが、理想的には以下のようなフレームワークにする:

```python
# 理想形: 宣言的な評価設定
# eval_config.yaml
test_suite:
  name: "japanese_singing_vc_v1"
  samples:
    - ref: "data/eval/ref_jpop_01.wav"
      conv: "data/eval/conv_jpop_01.wav"
      ground_truth_text: "きみのこえがきこえる"
      genre: "jpop"
      pitch_range: "mid"
    - ref: "data/eval/ref_enka_01.wav"
      conv: "data/eval/conv_enka_01.wav"
      genre: "enka"
      pitch_range: "low"

  metrics:
    - mcd:
        n_mfcc: 13
        config_file: "configs/v2/48k.json"
    - f0_rmse:
        method: "rmvpe"
    - whisper_cer:
        model: "medium"
        language: "ja"
    - speaker_similarity:
        model: "wespeaker"
    - formant_analysis:
        vowels: ["a", "i", "u", "e", "o"]

  thresholds:
    mcd: {pass: 6.0, warn: 8.0}
    f0_rmse: {pass: 20, warn: 50}
    whisper_cer: {pass: 0.10, warn: 0.20}
```

これにより:
- 新しいメトリクスの追加がプラグイン形式で可能
- テストスイートの定義が宣言的で、設定ファイルだけでテスト内容を把握可能
- 複数サンプルのバッチ評価、ジャンル別・音域別の集計が標準機能
- CI/CDパイプラインへの統合が容易

### 6.2 Whisper CER の代替・補完

Whisper CERの歌声信頼性問題に対して、理想的には:

1. **歌声特化ASR**: `NeMo` や `ESPnet` の日本語歌声認識モデルを使用。ただし現時点では十分な精度のモデルが存在しない
2. **強制アラインメント方式**: 正解テキストが既知の場合、`Montreal Forced Aligner` + `julius` で音素レベルのアラインメントを行い、音素正解率 (Phoneme Accuracy Rate) を計測する。CERよりも歌声に適した指標になる可能性がある
3. **UTMOS/PESQ**: 信号レベルの品質指標を追加して、転写精度に依存しない品質評価を実現
4. **A/B比較機能**: 2つの変換結果を並べて、各指標でどちらが優れているかを表示する比較モード

### 6.3 ベースライン管理

理想的なベースライン管理:

- **DVC (Data Version Control)** でテストデータと結果をバージョン管理
- **MLflow** や **Weights & Biases** で実験トラッキング。各マイルストーンの変更による指標の推移を可視化
- ベースラインの自動更新ルール（手動承認が必要な閾値を設定）

### 6.4 リアルタイム回帰テスト

理想的には、`infer-web.py` のWebUI上で変換を行うたびにバックグラウンドで評価スクリプトを実行し、品質劣化をリアルタイムで検出する仕組みを組み込む。Gradioの `after` コールバックで評価結果を表示パネルに反映する設計。

### 6.5 現実的な判断

上記の理想形に対して、M0では以下の判断でスコープを絞っている:

- フレームワーク化は将来の拡張性を意識しつつも、現時点では最小構成（CLIで1ペア評価）に留める
- 話者類似度は選定に時間がかかるためM0スコープ外とした
- ベースライン管理はJSONファイルで十分（DVC/MLflowは過剰）
- バッチ評価はシェルスクリプトのforループで代用可能

---

## 7. 後続タスクへの連絡事項

### M1 への引き継ぎ

1. **ベースライン値の参照方法**: M1-11「評価実行+ベースライン比較」では `tools/eval/baselines/baseline_contentvc.json` の値と比較する。`run_eval.py` の出力JSONとベースラインJSONのスキーマは同一であり、値の差分で改善量を定量化できる

2. **Go/No-Go判定の計算方法**:
   - MCD改善率: `(baseline_mcd - new_mcd) / baseline_mcd * 100` が 5% 以上で Go
   - F0 RMSE改善率: `(baseline_f0 - new_f0) / baseline_f0 * 100` が 10% 以上で Go
   - ベースライン値はM0で記録した `summary.mcd_mean` と `summary.f0_rmse_mean` を使用

3. **Whisper CERの解釈に関する注意**: 原音CER (`whisper_cer_ref_audio_mean`) が高い場合（例: 15%以上）、変換音声のCERが高くてもWhisperの歌声転写精度の問題であり、モデル品質の問題とは限らない。`conv_cer - ref_audio_cer` の差分が 5pt 以内であれば「発音明瞭度は原音と同等」と解釈する

4. **評価実行コマンド**:
   ```bash
   # 単一ペアの評価
   uv run python tools/eval/run_eval.py --ref ref.wav --conv conv.wav

   # 特定指標のみ
   uv run python tools/eval/run_eval.py --ref ref.wav --conv conv.wav --metrics mcd,f0

   # 結果をファイルに保存
   uv run python tools/eval/run_eval.py --ref ref.wav --conv conv.wav --output result.json
   ```

### M2 への引き継ぎ

1. **SSLモデル比較時の評価**: M2-6「kushinada vs ContentVec検証」では、同一テストデータに対して両モデルの変換結果を `run_eval.py` で評価し、CER差分が 5pt 以上であることを確認する

2. **話者類似度の追加**: M2以降で話者類似度メトリクスを追加する場合は、`tools/eval/metrics/speaker_similarity.py` として同じインターフェース（`compute_speaker_similarity(ref_path, conv_path) -> dict`）で実装すれば、`run_eval.py` への統合は最小限の変更で済む

3. **melパラメータの変更追従**: M3-Bで `mel_fmin` を `0.0 → 40.0` に変更する際、MCD計算のmelパラメータも合わせて変更が必要。`run_eval.py` の `--config` オプションで config ファイルを指定できるようにしておく

### 全マイルストーン共通

1. **評価結果の蓄積ルール**: 各マイルストーン完了時に、ベースラインJSONと同じフォーマットで結果を `tools/eval/baselines/` に保存する。命名規則: `baseline_<モデル名>_<日付>.json`（例: `baseline_kushinada_20260420.json`）

2. **新メトリクス追加時の手順**:
   - `tools/eval/metrics/` に新しいモジュールを作成
   - `compute_<metric_name>(ref_path, conv_path, **kwargs) -> dict` 関数を公開
   - `run_eval.py` の `THRESHOLDS` 辞書と `--metrics` 選択肢に追加
   - 対応するユニットテストを `tests/eval/` に追加

3. **テストデータの管理**: テスト用の歌声WAVファイルはリポジトリにコミットしない（サイズが大きいため）。パスと説明のみを `baselines/*.json` に記録し、実データは開発者間で共有する（Google Drive等）
