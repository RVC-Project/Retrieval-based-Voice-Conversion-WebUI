# 日本語歌声変換 マイルストーン計画

> 全ドキュメントを統合し策定（2026-04-11）
> レビュー反映版（5視点レビュー: 実現可能性/ROI/リスク/網羅性/ユーザー視点）

---

## プロジェクト概要

声優の歌声ボーカルデータ（数曲・約10分）から、日本語の発音精度が高い歌声変換モデルを構築する。
現行RVCの英語HuBERT（ContentVec）を日本語SSLモデルに置換し、歌声パイプライン全体を最適化する。

---

## 数値目標

| 指標 | 現状推定 | M1後 | M2後 | M3後 | 最終目標 |
|------|---------|------|------|------|---------|
| **Whisper CER（日本語明瞭度）** | 15-25% | 13-22% | 8-15% | 7-12% | **<10%** |
| **MCD（音色歪み）** | 7.5-8.5 dB | 7.0-8.0 dB | 6.0-7.0 dB | 5.5-6.5 dB | **<6.0 dB** |
| **F0 RMSE（ピッチ精度）** | 25-35 cents | 15-25 cents | 12-20 cents | 10-18 cents | **<20 cents** |
| **話者類似度** | 0.70-0.80 | 0.72-0.82 | 0.78-0.86 | 0.80-0.88 | **>0.80** |
| **主観MOS** | 3.0-3.5 | 3.2-3.7 | 3.5-4.0 | 3.8-4.2 | **>4.0** |

---

## マイルストーン一覧

```
M0+M1: 評価基盤 + 即効性改善 (Week 1-2, 並列実行)
  ↓ ← Go/No-Go判定①
M2: SSL置換+事前学習 (Week 3-5)
  ↓ ← Go/No-Go判定②
M3: 損失関数+ボコーダ改善 (Week 6-8)
  ↓ ← Go/No-Go判定③
M4: 高度な最適化 (Week 9+, オプション)
```

> **レビュー反映**: M0とM1は依存関係がなく並列実行可能。M1の設定変更は評価スクリプト完成前でも安全に適用できるため、Week 1-2で同時進行する。

---

## M0: 評価基盤構築

> **チケット**: [M0_evaluation_infrastructure.md](tickets/M0_evaluation_infrastructure.md)

**期間**: Week 1（3人日）
**GPU要件**: RTX 3060+（Whisper推論用）
**既存モデル互換性**: 影響なし

### 成果物

| # | タスク | 工数 | 成果物 |
|---|--------|------|--------|
| 0-1 | 評価スクリプト基盤作成 | 1日 | `tools/eval/run_eval.py`（CLI） |
| 0-2 | MCD + F0 RMSE 自動計測 | 0.5日 | `tools/eval/metrics/mcd.py`, `f0_accuracy.py` |
| 0-3 | Whisper CER パイプライン | 1日 | `tools/eval/metrics/whisper_cer.py`（日本語正規化込み） |
| 0-4 | ベースライン測定 | 0.5日 | 現行ContentVecモデルの各指標値を記録 |

### 追加パッケージ
```
openai-whisper, jiwer, jaconv, fastdtw
```

### 完了基準
- `uv run python tools/eval/run_eval.py --ref ref.wav --conv conv.wav` でJSON出力
- 現行モデルのベースライン値（MCD, F0 RMSE, CER）が記録済み

---

## M1: 即効性改善（既存モデル互換維持）

> **チケット**: [M1_immediate_improvements.md](tickets/M1_immediate_improvements.md)

**期間**: Week 1-2（10人日）※M0と並列実行
**GPU要件**: RTX 3060 12GB（既存環境で実施可能）
**既存モデル互換性**: **完全維持**

> **レビュー反映**: 工数を7→10人日に修正（テスト・デバッグ時間を加算）。MRSTFT損失をM3から前倒し（auraloss追加+数行で実装可能、互換性維持）。歌声プリセットをユーザー体験改善として追加。

### 成果物

| # | タスク | 工数 | 変更ファイル | 互換性 |
|---|--------|------|-------------|--------|
| 1-1 | FCPE統合（メインパイプライン） | 1日 | `pipeline.py` | 維持 |
| 1-2 | F0レンジ拡張（65-1400Hz） | 0.5日 | `pipeline.py`, `extract_f0_print.py` | 維持 |
| 1-3 | filter_radius デフォルト変更（3→1） | 0.5日 | `infer-web.py`, `infer_cli.py` | 維持 |
| 1-4 | Dropout(0.1) + Weight Decay(0.01) | 0.5日 | `configs/v2/*.json`, `train.py` | 維持 |
| 1-5 | segment_size拡張（17280→34560） | 0.5日 | `configs/v2/*.json` | 維持 |
| 1-6 | 歌唱向け前処理パラメータ | 1日 | `preprocess.py` | 維持 |
| ~~1-7~~ | ~~mel_fmin変更（0→40Hz）~~ | - | - | **M3-Bに延期**（既存事前学習モデルと非互換のため） |
| 1-8 | Multi-Resolution STFT損失追加 | 1日 | `losses.py`, `train.py` | 維持 |
| 1-9 | **歌声プリセット（WebUI）** | 1日 | `infer-web.py`, 新規`f0_presets.py` | 維持 |
| 1-10 | bfloat16移行 | 0.5日 | `configs/v2/*.json` | 維持 |
| 1-11 | 評価実行 + ベースライン比較 | 1日 | - | - |

> **レビュー反映**: データ拡張スクリプト(旧1-8)はM2-Bに移動。ターゲット話者FT時に使うものであり、M1段階では効果測定不可。代わりにMRSTFT損失・歌声プリセット・bfloat16をM1に追加。

### 歌声プリセット（タスク1-9）

WebUIに「歌声モード」ドロップダウンを追加し、ワンクリックで最適パラメータを適用:
```
J-POP:   rmvpe, filter_radius=1, f0_min=65, f0_max=1100
演歌:     rmvpe, filter_radius=0, f0_min=65, f0_max=1100（こぶし保存）
アニソン: fcpe, filter_radius=1, f0_min=80, f0_max=1400（広音域）
話し声:   rmvpe, filter_radius=3, f0_min=50, f0_max=800（従来互換）
```

### 追加パッケージ（M1）
```
auraloss>=0.4.0
```

### Go/No-Go判定① (Week 2終了時)

| 基準 | Go | No-Go対応 |
|------|-----|----------|
| MCD改善 | 5%以上改善 | パラメータ再調整、segment_size/filter_radius検証 |
| F0 RMSE改善 | 10%以上改善 | FCPEパラメータ、f0_min/f0_max微調整 |
| リアルタイムレイテンシ | 200ms以下を維持 | segment_size調整 |
| 既存モデル互換 | 完全維持 | - |

---

## M2: SSL置換 + 日本語歌声事前学習

> **チケット**: [M2A_ssl_model_integration.md](tickets/M2A_ssl_model_integration.md) | [M2B_pretrain_japanese_singing.md](tickets/M2B_pretrain_japanese_singing.md)

**期間**: Week 3-5（18.5人日 + GPU 14-30h）
**GPU要件**: RTX 4090 24GB（事前学習）、Cloud GPU（必要時）
**既存モデル互換性**: 新モデルのみ（旧モデルは互換モード維持）

### Phase 2-A: SSLモデル統合（Week 3, 10.5人日）

| # | タスク | 工数 | 変更ファイル |
|---|--------|------|-------------|
| 2-1 | SSLモデルローダー抽象化 | 2日 | `infer/modules/vc/utils.py`（新: `load_ssl_model()`） |
| 2-2 | HuggingFace transformers ラッパー | 1日 | 新規モジュール |
| 2-3 | 推論パイプライン対応（output_layer設定化） | 1日 | `pipeline.py`, `rtrvc.py` |
| 2-4 | 学習パイプライン対応 | 1日 | `extract_feature_print.py` |
| 2-5 | モデル定義のssl_dimパラメータ化 | **2日** | `models.py`（768ハードコードの解消+互換性ロジック） |
| 2-6 | kushinada vs ContentVec 検証 | 0.5日 | - |
| 2-6b | **WebUI SSLモデル選択UI** | 1日 | `infer-web.py` |
| 2-6c | **Weighted Sum of Layers導入** | 2日 | `pipeline.py`, `models.py` |

#### SSLモデル統合方針

> **レビュー反映**: 4モデル比較実験（2.5人日）は過剰。kushinada一本に絞り、問題発生時のみrinnaにフォールバックする戦略に変更。比較実験を2.5人日→0.5人日に縮小。

**第1候補**: `imprt/kushinada-hubert-base`（62,215h日本語、768次元、Apache 2.0）
**フォールバック**: `rinna/japanese-hubert-base`（19,000h、nadare氏の事前学習チェックポイントが利用可能）

検証: kushinada + ContentVec（ベースライン）の2モデルのみでCER比較。CER 5pt以上改善が確認できなければrinnaを試行。

### Phase 2-B: 日本語歌声事前学習（Week 4-5, 8人日 + GPU 14-30h）

| # | タスク | 工数 | GPU時間 |
|---|--------|------|---------|
| 2-7 | データセット準備（JVS-MuSiC 24k→48kリサンプル等） | 2日 | ※24k→48kは高域欠損に注意 |
| 2-7b | **データ拡張スクリプト（ピッチシフト±4半音）** | 1.5日 | 新規スクリプト |
| 2-8 | 多話者filelist生成スクリプト | **1.5日** | spk_embed_dim変更含む。タスク2-4完了が前提 |
| 2-9 | Stage 1: JVS-MuSiC 100話者事前学習 | 1日 | ~4h (RTX 4090) |
| 2-10 | Stage 2: 歌声DB適応（NIT+きりたん+GTSinger） | 1日 | ~10h (RTX 4090) |
| 2-11 | ContentVec版 + kushinada版の2バリアント作成 | 1.5日 | ~14h |
| 2-12 | ターゲット話者ファインチューニング + 品質評価 | 2日 | ~2h |

#### 事前学習データセット

| データセット | 話者数 | 時間 | ライセンス | Stage |
|-------------|--------|------|-----------|-------|
| JVS-MuSiC | 100 | ~3.3h | 商用可 | Stage 1 |
| NIT-SONG070 | 1 | 1.2h | CC BY 3.0 | Stage 2 |
| GTSinger(JA) | 2 | ~8h | 研究OS | Stage 2 |
| 東北きりたん | 1 | 57min | 研究のみ | Stage 2 |
| 東北イタコ | 1 | ~1h | 研究のみ | Stage 2 |

### Go/No-Go判定② (Week 5終了時)

| 基準 | Go | No-Go対応 |
|------|-----|----------|
| CER改善 | kushinada/rinnaがContentVec比で**CER 5pt以上改善** | rinnaチェックポイント(BOOTH)で代替 |
| 事前学習効果 | ファインチューニング収束が3-5エポックに短縮 | エポック数調整、学習率変更 |
| 歌声品質 | 主観評価でベースライン以上 | SSLモデル候補の再検討 |

---

## M3: 損失関数 + ボコーダ改善

> **チケット**: [M3A_loss_function_improvements.md](tickets/M3A_loss_function_improvements.md) | [M3B_vocoder_improvements.md](tickets/M3B_vocoder_improvements.md)

**期間**: Week 6-8（11人日 + GPU 数日）
**GPU要件**: RTX 4090 24GB / Cloud GPU（SnakeBeta導入時に事前学習再実行）
**既存モデル互換性**: Phase 3-Aは維持、Phase 3-Bは事前学習再実行必要

### Phase 3-A: 損失関数改善（Week 6, 4人日）

| # | タスク | 工数 | 互換性 |
|---|--------|------|--------|
| 3-1 | Multi-Resolution STFT損失（検証・チューニング） | 0.5日 | 維持 |
| 3-2 | KLサイクリカルアニーリング | 0.5日 | 維持 |
| 3-3 | EMA（alpha=0.999） | 1日 | 維持 |
| 3-4 | CosineAnnealingWarmRestarts + Warmup | 1日 | 維持 |
| 3-5 | DWTビブラート保存 | 1日 | 維持 |

> **注意**: タスク3-1のMRSTFT損失の実装自体はM1（タスク1-8）で完了済み。M3-Aでは係数チューニングのみ。`auraloss>=0.4.0` もM1で追加済み。

追加パッケージ: `PyWavelets>=1.4.0`

### Phase 3-B: ボコーダ改善（Week 7-8, 7人日）

| # | タスク | 工数 | 互換性 |
|---|--------|------|--------|
| 3-6 | SnakeBeta活性化関数導入 | **3-4日** | **事前学習再実行**。mel_fmin=40Hzもここで適用 |
| 3-7 | アンチエイリアスフィルタ追加 | 2日 | **事前学習再実行** |
| 3-8 | MRD/CQTディスクリミネータ追加 | 2日 | 学習時のみ |
| 3-9 | 事前学習再実行（kushinada + SnakeBeta） | 1日 | GPU 14-30h |

### Go/No-Go判定③ (Week 8終了時)

| 基準 | Go(M4へ) | M3で完了 |
|------|----------|---------|
| MCD | <6.5 dB | M3成果で十分と判断 |
| CER | <12% | 主観品質が十分 |
| MOS | >3.8 | >3.5で実用レベル |

---

## M4: 高度な最適化（オプション）

> **チケット**: [M4_advanced_optimization.md](tickets/M4_advanced_optimization.md)

**期間**: Week 9+（数ヶ月単位）
**GPU要件**: A100 40GB+推奨
**前提**: M1-M3の成果が確認済み

### 候補タスク（効果/コスト順）

| # | タスク | 工数 | GPU要件 | 期待効果 |
|---|--------|------|---------|---------|
| 4-1 | F0量子化ビン拡大（256→512） | 3日 | 事前学習再実行 | ビブラート精度向上 |
| 4-2 | 重み付き和（Weighted Sum of Layers） | 2日 | - | SSL特徴量最適化 |
| 4-3 | kNN-SVC加算合成（倍音強化） | 3-4週間 | RAM 16GB+ | 音色リッチさ向上 |
| 4-4 | BigVGAN / HiFTNet完全移行 | 4-6週間 | RTX 4090+ | 高域品質大幅向上 |
| 4-5 | Shallow Diffusion後段追加 | 4-6週間 | A100 | UTMOS +0.5-0.8 |
| 4-6 | TTS逆翻訳データ拡張（GPT-SoVITS） | 2-3週間 | RTX 3090+ | 実効データ3-5倍 |
| 4-7 | 適応的index_rate | 3-5日 | 現行同等 | 子音/無声区間改善 |
| 4-8 | GRPO強化学習 | 6-8週間 | 4xA100 | MOS最適化 |

---

## 依存関係マップ（レビュー反映版）

```
Week 1-2: M0+M1 並列実行
 ┌── M0 [評価基盤] ──────────────────┐
 │    ├── 評価スクリプト              │
 │    └── ベースライン測定            │
 │                                  │
 └── M1 [即効性改善] ───────────────┤  ← 並列実行可能
      ├── FCPE統合                  │
      ├── F0レンジ拡張               │
      ├── Dropout/WD/bfloat16       │
      ├── segment_size拡張          │
      ├── 歌唱前処理                 │
      ├── MRSTFT損失（M3から前倒し）  │
      └── 歌声プリセット（WebUI）     │
                                    │
Week 3-5: M2                        │
 M2 [SSL置換+事前学習] ←────────────┘
   ├── SSLローダー抽象化
   ├── kushinada統合+検証 ←── M0必須
   ├── WebUI SSLモデル選択UI
   ├── Weighted Sum of Layers
   ├── データ拡張スクリプト（M1から移動）
   ├── 日本語歌声事前学習
   │     ├── Stage 1: JVS-MuSiC
   │     └── Stage 2: 歌声DB適応
   └── ターゲット話者FT + 評価
            │
Week 6-8: M3
   ├──→ M3-A [損失関数改善]
   │     ├── KLアニーリング
   │     ├── EMA
   │     └── DWTビブラート
   │
   └──→ M3-B [ボコーダ改善]
         ├── SnakeBeta ──→ 事前学習再実行
         ├── アンチエイリアス
         └── MRD/CQT
               │
               └──→ M4 [高度な最適化]
```

### 並列化可能なタスク

| グループ | タスク | 担当リソース |
|---------|--------|------------|
| A（CPU/スクリプト） | 評価基盤、前処理最適化、データ拡張スクリプト | 開発者1 |
| B（コード変更） | FCPE統合、F0改善、Dropout追加 | 開発者2 |
| C（GPU計算） | SSLモデル比較実験、事前学習 | GPU 1台 |
| D（リサーチ） | ベンチマーク測定、A/Bテスト設計 | 開発者1 or 3 |

---

## ハードウェア要件サマリー

### 実際のハードウェア構成

| 用途 | GPU | VRAM | 備考 |
|------|-----|------|------|
| 開発・実装・推論テスト | **RTX 4070 Ti Super** | 16GB | メインPC |
| ファインチューニング・学習 | **RTX 4090** | 24GB | 学習専用1台 |
| 事前学習（M2/M3） | **Cloud GPU** | 必要に応じ | RTX 4090で不足時のみ |

### マイルストーン別GPU割り当て

| マイルストーン | 使用GPU | 備考 |
|-------------|---------|------|
| M0+M1 | RTX 4070 Ti Super (16GB) | 開発・評価・推論すべてローカルで完結 |
| M2（SSL統合・検証） | RTX 4070 Ti Super (16GB) | 実装・推論テスト |
| M2（事前学習） | RTX 4090 (24GB) | batch_size調整で対応可。不足時はCloud GPU |
| M3-A（損失関数） | RTX 4070 Ti Super (16GB) | 実装のみ |
| M3-B（ボコーダ+事前学習再実行） | RTX 4090 (24GB) / Cloud GPU | SnakeBeta導入後の再学習 |
| M4 | Cloud GPU（A100推奨） | 大規模実験のみ |

### エンドユーザー（ファインチューニング・推論）

| 用途 | 最小GPU | 推奨GPU |
|------|--------|---------|
| ファインチューニング（10分データ） | RTX 3060 12GB | RTX 3060 12GB |
| 推論（オフライン） | RTX 2060 6GB | RTX 3060 12GB |
| リアルタイム変換 | RTX 3060 12GB | RTX 4060 8GB |

> 事前学習はRTX 4090またはCloud GPUで実施。エンドユーザーは事前学習済みチェックポイントを利用するため、RTX 3060で十分。

---

## リスク管理

> **レビュー反映**: 実現可能性・リスク・網羅性レビューの指摘を統合。致命的リスク5件を追加。

### 致命的リスク

| リスク | 影響度 | 発生確率 | 対策 |
|--------|--------|---------|------|
| **kushinadaが歌声で効果不足**（TV放送話声で学習、歌声ドメインミスマッチ） | 致命的 | **高** | M1期間中にkushinadaの歌声予備実験を実施。Spin V2も比較候補に追加。最悪はMERT（歌声MOS最高3.72）を検討 |
| ~~CC BY-SAの汚染効果~~ | ~~致命的~~ | - | **解消済み**: PJS（CC BY-SA 4.0）をデータセットから除外。使用データはすべて商用可またはCC BYライセンス |
| **ロールバック計画の欠如** | 致命的 | N/A | 各MS開始前にgitタグ。config変更は`48k_singing.json`として新規作成（既存config変更禁止）。事前学習モデルは命名規則で分離 |
| **mel_fmin変更の互換性問題** | 高 | 確実 | mel_fmin変更は既存事前学習モデルと非互換。**M1では変更せず、M3-B事前学習再実行時にまとめて適用** |
| **fairseq依存の非互換** | 致命的 | 中 | PyTorch 2.10との互換性を事前検証。M2でtransformers移行を最優先 |

### 高リスク

| リスク | 影響度 | 発生確率 | 対策 |
|--------|--------|---------|------|
| segment_size倍増でOOM | 中 | 中 | RTX 4070 Ti Super 16GBならbatch_size=4で対応可能な見込み。RTX 4090 24GBでは問題なし |
| JVS-MuSiC 24→48kHz高域欠損 | 高 | **確実** | Stage 1ではmel_fmax=12kHzに制限。または32kHz configで事前学習 |
| Whisper CERの歌声信頼性不足 | 高 | 高 | M0で原音CERをベースライン測定。開発者3名の簡易MOSを併用 |
| 多話者filelist生成の隠れた複雑性 | 中 | 高 | 工数0.5日→1.5日に修正。タスク2-4完了を前提条件化 |
| spk_embed_dim変更の互換性 | 高 | 確実 | 事前学習用configを別ファイル(`48k_pretrain.json`)で管理 |

### 中リスク

| リスク | 影響度 | 発生確率 | 対策 |
|--------|--------|---------|------|
| 事前学習のVRAM不足 | 中 | 低 | batch_size削減、クラウドGPU（コスト$150-250） |
| 学習データ量の不足（10分） | 高 | 中 | データ拡張で70-100分相当に拡大 |
| 過学習 | 中 | 高 | Dropout(0.1)+WD(0.01)+EMA+SpecAugment |
| リアルタイムレイテンシ劣化 | 中 | 低 | Go/No-Go判定にレイテンシ計測追加（200ms閾値） |
| Windows環境での分散学習問題 | 中 | 高 | WSL2の利用を検討。単一GPU学習を優先 |

### ロールバック手順

| マイルストーン | ロールバック方法 |
|-------------|----------------|
| M1 | gitタグ `pre-m1` にrevert。既存configは変更しないため安全 |
| M2 | `feature/ssl-abstraction` ブランチで作業。mainは安定版維持 |
| M3-B | M2チェックポイントに戻す。SnakeBeta前のコードはgitタグで保存 |

---

## MVP（最小実行可能改善）

> **レビュー反映**: 2-3日で出荷可能な最小改善セット。評価スクリプトなしでも聴覚テストで効果確認可能。

以下は**設定値の変更のみ**であり、既存モデルとの互換性を完全に維持する:

| 変更 | ファイル | 所要時間 |
|------|---------|---------|
| `p_dropout: 0` → `0.1` | `configs/v2/48k.json`, `32k.json` | 5分 |
| `mel_fmin: 0.0` → `40.0` | 同上 | 5分 |
| `segment_size: 17280` → `34560` | 同上 | 5分 |
| `filter_radius` デフォルト `3` → `1` | `infer_cli.py`, `infer-web.py` | 30分 |
| `f0_min: 50` → `65` | `pipeline.py` | 30分 |

**これだけで再学習すれば、ピッチ精度・過学習防止・ビブラート保存が改善される。**

---

## エンドユーザー向け成果物（各マイルストーン）

> **レビュー反映**: 各マイルストーンでユーザーが体感できる改善を明示。

| MS | ユーザーが体感できる改善 | 既存モデル |
|----|----------------------|-----------|
| M1 | 歌声プリセット（J-POP/演歌/アニソン）、高音域の音程精度向上、ビブラート保存 | **そのまま使える** |
| M2 | 日本語の歌詞が聞き取りやすくなる、学習が速くなる（3-5エポック） | 再学習が必要 |
| M3 | 音質の大幅向上（子音の鋭さ、高域の透明感） | 再学習が必要 |

### ドキュメント計画

| MS | 作成するドキュメント |
|----|-------------------|
| M1 | 歌声モデル作成クイックスタートガイド（データ準備→学習→推論） |
| M2 | SSLモデル選択ガイド（WebUIでの操作手順） |
| M3 | 推奨パラメータ設定一覧 |

---

## 関連ドキュメント

### チケット（実装詳細）

| チケット | 内容 |
|---------|------|
| [tickets/README.md](tickets/README.md) | **チケット一覧・進捗サマリー** |
| [tickets/M0_evaluation_infrastructure.md](tickets/M0_evaluation_infrastructure.md) | M0: 評価基盤構築 |
| [tickets/M1_immediate_improvements.md](tickets/M1_immediate_improvements.md) | M1: 即効性改善 |
| [tickets/M2A_ssl_model_integration.md](tickets/M2A_ssl_model_integration.md) | M2-A: SSLモデル統合 |
| [tickets/M2B_pretrain_japanese_singing.md](tickets/M2B_pretrain_japanese_singing.md) | M2-B: 日本語歌声事前学習 |
| [tickets/M3A_loss_function_improvements.md](tickets/M3A_loss_function_improvements.md) | M3-A: 損失関数改善 |
| [tickets/M3B_vocoder_improvements.md](tickets/M3B_vocoder_improvements.md) | M3-B: ボコーダ改善 |
| [tickets/M4_advanced_optimization.md](tickets/M4_advanced_optimization.md) | M4: 高度な最適化 |

### 調査・提案書

| ドキュメント | 内容 |
|-------------|------|
| [requirements_japanese_singing_vc.md](requirements_japanese_singing_vc.md) | 要求定義書 |
| [research_japanese_singing_vc.md](research_japanese_singing_vc.md) | 技術調査レポート（10エージェント） |
| [research_ssl_models_update.md](research_ssl_models_update.md) | SSLモデル追加調査（5エージェント） |
| [datasets_japanese_singing.md](datasets_japanese_singing.md) | 日本語歌声データセット一覧 |
| [proposals_summary.md](proposals_summary.md) | 品質向上提案書10件の統合サマリー |
| [proposal_data_strategy.md](proposal_data_strategy.md) | データ戦略の詳細 |
| [proposal_segment_length_architecture.md](proposal_segment_length_architecture.md) | セグメント長・モデル構造の詳細 |
| [proposal_vocoder_decoder_improvements.md](proposal_vocoder_decoder_improvements.md) | ボコーダ改善の詳細 |
