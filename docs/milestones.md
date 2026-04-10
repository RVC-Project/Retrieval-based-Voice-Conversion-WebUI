# 日本語歌声変換 マイルストーン計画

> 全ドキュメントを統合し策定（2026-04-11）

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
M0: 評価基盤構築 (Week 1)
  ↓
M1: 即効性改善 (Week 1-2)
  ↓ ← Go/No-Go判定①
M2: SSL置換+事前学習 (Week 3-5)
  ↓ ← Go/No-Go判定②
M3: 損失関数+ボコーダ改善 (Week 6-8)
  ↓ ← Go/No-Go判定③
M4: 高度な最適化 (Week 9+, オプション)
```

---

## M0: 評価基盤構築

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

**期間**: Week 1-2（7人日）
**GPU要件**: RTX 3060 12GB（既存環境で実施可能）
**既存モデル互換性**: **完全維持**

### 成果物

| # | タスク | 工数 | 変更ファイル | 互換性 |
|---|--------|------|-------------|--------|
| 1-1 | FCPE統合（メインパイプライン） | 1日 | `pipeline.py` | 維持 |
| 1-2 | F0レンジ拡張（65-1400Hz） | 0.5日 | `pipeline.py`, `extract_f0_print.py` | 維持 |
| 1-3 | filter_radius デフォルト変更（3→1） | 0.5日 | `infer-web.py`, `infer_cli.py` | 維持 |
| 1-4 | Dropout(0.1) + Weight Decay(0.01) | 0.5日 | `configs/v2/*.json`, `train.py` | 維持 |
| 1-5 | segment_size拡張（17280→34560） | 0.5日 | `configs/v2/*.json` | 維持 |
| 1-6 | 歌唱向け前処理パラメータ | 1日 | `preprocess.py` | 維持 |
| 1-7 | mel_fmin変更（0→40Hz） | 0.5日 | `configs/v2/*.json` | 維持 |
| 1-8 | データ拡張スクリプト（ピッチシフト±4半音） | 1.5日 | 新規スクリプト | - |
| 1-9 | 評価実行 + ベースライン比較 | 1日 | - | - |

### Go/No-Go判定① (Week 2終了時)

| 基準 | Go | No-Go対応 |
|------|-----|----------|
| MCD改善 | 5%以上改善 | パラメータ再調整、segment_size/filter_radius検証 |
| F0 RMSE改善 | 10%以上改善 | FCPEパラメータ、f0_min/f0_max微調整 |
| 既存モデル互換 | 完全維持 | - |

---

## M2: SSL置換 + 日本語歌声事前学習

**期間**: Week 3-5（16人日 + GPU 14-30h）
**GPU要件**: RTX 3090 24GB
**既存モデル互換性**: 新モデルのみ（旧モデルは互換モード維持）

### Phase 2-A: SSLモデル統合（Week 3, 8人日）

| # | タスク | 工数 | 変更ファイル |
|---|--------|------|-------------|
| 2-1 | SSLモデルローダー抽象化 | 2日 | `infer/modules/vc/utils.py`（新: `load_ssl_model()`） |
| 2-2 | HuggingFace transformers ラッパー | 1日 | 新規モジュール |
| 2-3 | 推論パイプライン対応（output_layer設定化） | 1日 | `pipeline.py`, `rtrvc.py` |
| 2-4 | 学習パイプライン対応 | 1日 | `extract_feature_print.py` |
| 2-5 | モデル定義のssl_dimパラメータ化 | 0.5日 | `models.py` |
| 2-6 | SSLモデル比較実験 | 2.5日 | - |

#### SSLモデル比較実験の構成

| モデル | 日本語データ | 備考 |
|--------|------------|------|
| **imprt/kushinada-hubert-base** | 62,215h | **最有力候補** |
| reazon-research/japanese-wav2vec2-base | 35,000h+ | 次点候補 |
| rinna/japanese-hubert-base | 19,000h | 既存実績あり |
| ContentVec（現行） | 英語960h | ベースライン |

比較指標: Whisper CER + MCD + 話者類似度（M0で構築済みの評価パイプライン使用）

### Phase 2-B: 日本語歌声事前学習（Week 4-5, 8人日 + GPU 14-30h）

| # | タスク | 工数 | GPU時間 |
|---|--------|------|---------|
| 2-7 | データセット準備（JVS-MuSiC 24k→48kリサンプル等） | 2日 | - |
| 2-8 | 多話者filelist生成スクリプト | 0.5日 | - |
| 2-9 | Stage 1: JVS-MuSiC 100話者事前学習 | 1日 | ~4h (RTX 3090) |
| 2-10 | Stage 2: 歌声DB適応（PJS+NIT+きりたん+GTSinger） | 1日 | ~10h (RTX 3090) |
| 2-11 | ContentVec版 + kushinada版の2バリアント作成 | 1.5日 | ~14h |
| 2-12 | ターゲット話者ファインチューニング + 品質評価 | 2日 | ~2h |

#### 事前学習データセット

| データセット | 話者数 | 時間 | ライセンス | Stage |
|-------------|--------|------|-----------|-------|
| JVS-MuSiC | 100 | ~3.3h | 商用可 | Stage 1 |
| PJS | 1 | ~30min | CC BY-SA 4.0 | Stage 2 |
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

**期間**: Week 6-8（11人日 + GPU 数日）
**GPU要件**: RTX 3090 24GB（SnakeBeta導入時に事前学習再実行）
**既存モデル互換性**: Phase 3-Aは維持、Phase 3-Bは事前学習再実行必要

### Phase 3-A: 損失関数改善（Week 6, 4人日）

| # | タスク | 工数 | 互換性 |
|---|--------|------|--------|
| 3-1 | Multi-Resolution STFT損失追加 | 0.5日 | 維持 |
| 3-2 | KLサイクリカルアニーリング | 0.5日 | 維持 |
| 3-3 | EMA（alpha=0.999） | 1日 | 維持 |
| 3-4 | CosineAnnealingWarmRestarts + Warmup | 1日 | 維持 |
| 3-5 | DWTビブラート保存 | 1日 | 維持 |

追加パッケージ: `auraloss>=0.4.0`, `PyWavelets>=1.4.0`

### Phase 3-B: ボコーダ改善（Week 7-8, 7人日）

| # | タスク | 工数 | 互換性 |
|---|--------|------|--------|
| 3-6 | SnakeBeta活性化関数導入 | 2日 | **事前学習再実行** |
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

## 依存関係マップ

```
M0 [評価基盤]
 │
 ├──→ M1 [即効性改善] ──────────────────────────┐
 │     ├── FCPE統合                              │
 │     ├── F0レンジ拡張                           │
 │     ├── Dropout/WD                            │
 │     ├── segment_size拡張                      │
 │     ├── 歌唱前処理                             │
 │     └── データ拡張スクリプト                      │
 │                                               │
 └──→ M2 [SSL置換+事前学習]                       │
       ├── SSLローダー抽象化                       │
       ├── kushinada/rinna比較実験 ←── M0必須      │
       ├── 日本語歌声事前学習                       │
       │     ├── Stage 1: JVS-MuSiC              │
       │     └── Stage 2: 歌声DB適応              │
       └── ターゲット話者FT + 評価                  │
                │                                │
                ├──→ M3-A [損失関数改善] ←─────────┘
                │     ├── MRSTFT損失
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

| マイルストーン | 最小GPU | 推奨GPU | ストレージ | RAM |
|-------------|--------|---------|----------|------|
| M0 | RTX 3060 12GB | RTX 3060 12GB | 5GB | 16GB |
| M1 | RTX 3060 12GB | RTX 3060 12GB | 10GB | 16GB |
| M2 | RTX 3090 24GB | RTX 4090 24GB | 50GB | 32GB |
| M3 | RTX 3090 24GB | A100 40GB | 50GB | 32GB |
| M4 | A100 40GB | A100 80GB | 100GB+ | 64GB |

---

## リスク管理

| リスク | 影響度 | 発生確率 | 対策 |
|--------|--------|---------|------|
| kushinadaが歌声で効果不足 | 高 | 中 | rinna(実績あり)+nadareチェックポイントにフォールバック |
| 事前学習のVRAM不足 | 中 | 低 | batch_size削減、gradient checkpointing、クラウドGPU |
| 既存モデル互換性の破壊 | 高 | 中 | M1は互換維持必須。M2以降は互換モード(ssl_model設定)で対応 |
| データセットライセンス制約 | 中 | 中 | 商用パス(NIT+PJS+JVS)と研究パス(+きりたん+GTSinger)を分離 |
| 学習データ量の不足（10分） | 高 | 中 | データ拡張で70-100分相当に拡大。カリキュラム学習で効率化 |
| 過学習 | 中 | 高 | Dropout(0.1)+WD(0.01)+EMA+SpecAugment+Early Stopping |

---

## 関連ドキュメント

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
