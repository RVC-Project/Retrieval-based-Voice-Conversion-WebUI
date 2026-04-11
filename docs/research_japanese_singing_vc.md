# 日本語歌声変換モデル品質向上 調査レポート

> 10エージェント並列調査による包括的技術調査（2026-04-10）

---

## 目次

1. [エグゼクティブサマリー](#1-エグゼクティブサマリー)
2. [日本語SSLモデル（HuBERT代替）](#2-日本語sslモデルhubert代替)
3. [日本語音声学的分析と現行モデルの課題](#3-日本語音声学的分析と現行モデルの課題)
4. [歌声変換SOTA技術動向](#4-歌声変換sota技術動向)
5. [F0・ピッチ処理の改善](#5-f0ピッチ処理の改善)
6. [損失関数・学習手法の改善](#6-損失関数学習手法の改善)
7. [ボコーダ・デコーダの改善](#7-ボコーダデコーダの改善)
8. [データ拡張・前処理](#8-データ拡張前処理)
9. [RVCコミュニティの知見](#9-rvcコミュニティの知見)
10. [評価手法](#10-評価手法)
11. [統合改善ロードマップ](#11-統合改善ロードマップ)

---

## 1. エグゼクティブサマリー

### 最大のボトルネック

RVCの日本語歌声品質を制限する最大の要因は、**英語のみで学習されたHuBERT base（ContentVec）を特徴量抽出器として使用している点**である。LibriSpeech 960hの英語音声で学習されたこのモデルは、日本語の5母音体系・モーラ拍・ピッチアクセントを正確にエンコードできず、FAISS検索の品質にも直接影響する。

### 改善効果の見込み（優先度順）

| 優先度 | 改善施策 | 期待効果 | 実装難易度 | 状態 |
|--------|---------|---------|-----------|------|
| **P0** | 日本語HuBERT導入 + 事前学習チェックポイント | 日本語明瞭度30-50%改善 | 中 | 未実装 |
| **P0** | FCPEのメインパイプライン統合 | ピッチ精度向上+高速化 | 低 | ✅ 実装済み |
| **P0** | 評価パイプライン構築 | 品質の定量測定 | 中 | ✅ 実装済み |
| **P1** | 歌声向け前処理パイプライン | データ品質向上→全体品質向上 | 低 | ✅ 実装済み |
| **P1** | Multi-Resolution STFT損失追加 | 音質改善（過渡特性+定常構造） | 低 | ✅ 実装済み |
| **P1** | 正則化 (dropout, weight decay) | 過学習防止 | 低 | ✅ 実装済み |
| **P1** | KLアニーリング導入 | 潜在空間の有効活用 | 低 | 未実装 |
| **P2** | SnakeBeta活性化関数 + アンチエイリアス | 高音域アーティファクト削減 | 中 | 未実装 |
| **P2** | データ拡張（ピッチシフト+TTS） | 実効データ量4-5倍化 | 中 | 未実装 |
| **P2** | セグメント長拡張 | 歌声フレーズの文脈学習 | 中 | ✅ 実装済み |
| **P3** | MRD/CQTディスクリミネータ追加 | ハーモニック構造の再現向上 | 中 | 未実装 |
| **P3** | 浅い拡散モデルの後段追加 | 品質の最終ポリッシュ | 高 | 未実装 |

---

## 2. 日本語SSLモデル（HuBERT代替）

### 2.1 候補モデル比較

| モデル | 開発元 | 学習データ | 出力次元 | 層数 | 日本語対応 | RVC互換性 | ライセンス |
|--------|--------|-----------|---------|------|-----------|----------|-----------|
| **ContentVec（現行）** | MIT | LibriSpeech 960h（英語） | 768 | 12 | なし | ネイティブ | MIT |
| **imprt/kushinada-hubert-base** | 産総研 | TV放送62,215h | 768 | 12 | **ネイティブ** | **完全互換** | Apache 2.0 |
| **rinna/japanese-hubert-base** | rinna | ReazonSpeech 19,000h（日本語） | 768 | 12 | **ネイティブ** | **高**（768次元一致） | Apache 2.0 |
| **rinna/japanese-hubert-large** | rinna | ReazonSpeech 19,000h（日本語） | 1024 | 24 | **ネイティブ** | 要次元調整 | Apache 2.0 |
| **reazon-research/japanese-wav2vec2-base** | Reazon | ReazonSpeech v2.0 35,000h+ | 768 | 12 | **ネイティブ** | 高 | Apache 2.0 |
| **rinna/japanese-data2vec-audio-base** | rinna | ReazonSpeech v1 19,000h | 768 | 12 | **ネイティブ** | 高 | Apache 2.0 |
| **rinna/japanese-wav2vec2-base** | rinna | ReazonSpeech v1 19,000h | 768 | 12 | **ネイティブ** | 高 | Apache 2.0 |
| **sarulab-speech/hubert-base-jtube** | 猿渡研(東大) | YouTube 2,720h | 768 | 12 | **ネイティブ** | 高 | MIT |
| **Spin V2** | Applio | ContentVec微調整 | 768 | 12 | - | 高 | - |
| **WavLM Large** | Microsoft | 多言語94,000h | 1024 | 24 | 含む | 要次元調整 | MIT |
| **XLS-R (wav2vec 2.0)** | Meta | 128言語436,000h | 1024 | 24/48 | 含む | 要次元調整 | MIT |
| **XLSR-53** | Meta | 53言語56,000h | 1024 | 24 | 含む | 要次元調整 | MIT |
| **mHuBERT-147** | UTTER | 147言語 | 768 | 12 | 含む | 高（768次元一致） | MIT |
| **MERT** | 音楽特化 | 音楽データ | 768/1024 | 12/24 | 言語非依存 | 中 | CC BY-NC-SA 4.0 |
| **Whisper Encoder** | OpenAI | 680,000h多言語 | 512/1024 | 12-32 | **高品質** | 要次元調整 | MIT |

### 2.2 最有力候補: imprt/kushinada-hubert-base（産総研）

**推奨理由:**
- **62,215時間の日本語TV放送データ**: rinnaの19,000hの約3.3倍の大規模日本語音声で学習済み
- **768次元出力**: RVCのTextEncoder入力と完全一致。モデル変更不要
- **12層Transformer**: HuBERT baseと同一アーキテクチャ。ドロップイン置換可能
- **Apache 2.0ライセンス**: 商用利用可能
- **SERベンチマーク**: 84.77%（従来最高 70.65% を大幅超過）
- **2025年3月発表**: 産総研（AIST）による最新モデル

**技術仕様:**
- アーキテクチャ: HuBERT Base（12 Transformer層、12 attention heads）
- 出力: `last_hidden_state` サイズ `[1, #frames, 768]`
- 入力: 16kHzモノラル音声
- 学習データ: 日本語TV放送 62,215時間
- HuggingFace: https://huggingface.co/imprt/kushinada-hubert-base

**注意点:**
- ContentVecと異なり話者情報の除去処理がされていないため、embeddingに話者情報が多く含まれる可能性あり
- **対策**: 推論時のindex ratioを0.8-1.0に引き上げて話者情報の影響を補正

### 2.3 次点候補: rinna/japanese-hubert-base

**推奨理由:**
- **768次元出力**: RVCのTextEncoder入力と完全一致。モデル変更不要
- **12層Transformer**: HuBERT baseと同一アーキテクチャ。ドロップイン置換可能
- **ReazonSpeech 19,000時間**: 大規模日本語音声で学習済み
- **Apache 2.0ライセンス**: 商用利用可能
- **nadare氏の実証**: Qiita記事で日本語発音の明瞭度向上が確認済み
- **事前学習チェックポイント存在**: nadare氏によるRVC用事前学習済みチェックポイントが公開されており、すぐに利用可能

**技術仕様:**
- アーキテクチャ: HuBERT Base（12 Transformer層、12 attention heads）
- 出力: `last_hidden_state` サイズ `[1, #frames, 768]`
- 入力: 16kHzモノラル音声
- モデルサイズ: 約94.4MB
- HuggingFace: https://huggingface.co/rinna/japanese-hubert-base

**注意点:**
- ContentVecと異なり話者情報の除去処理がされていないため、embeddingに話者情報が多く含まれる
- **対策**: 推論時のindex ratioを0.8-1.0に引き上げて話者情報の影響を補正

### 2.4 nadare氏の事前学習済みチェックポイント

nadare氏（Qiita: https://qiita.com/nadare/items/18cd74e51c731904c3b0）がRVCモデルを日本語向けに事前学習し、以下のチェックポイントを公開している。

| チェックポイント | Phone Embedder | 特徴 |
|----------------|---------------|------|
| `f0X48k768_contentvec_v2` | ContentVec | 安定した変換。日本語発音はやや弱い |
| `f0X48k768_jphubert_v2` | rinna/jp-hubert-base | **日本語発音がクリア**。index ratio 0.8-1.0必須 |

- **配布先**: BOOTH（https://booth.pm/ja/items/4802383）、Kaggle
- **学習設定**: RTX 3090、バッチサイズ32、10エポック、約3時間
- **学習データ**: CSTR VCTKコーパス（48kHz、108話者、約50時間）

### 2.5 歌声特化SSL: MERT

音楽理解に特化したSSLモデル（ICLR 2024）。RVCのHuBERT代替として検討価値あり。

- RVQ-VAEベースの音響教師 + CQT（定Q変換）ベースの音楽教師のマルチタスク学習
- ビート・ピッチ・ローカル音色（歌手情報）のタスクで優れた性能
- 95M-330Mパラメータにスケーリング可能
- GitHub: https://github.com/yizhilll/MERT

### 2.6 層選択の研究知見

- **現行**: layer 12固定（HuBERTの最終層）
- **LinearVC研究（Interspeech 2025）**: WavLM-Largeの**layer 6**が音声変換に最適と確認。浅い層ほど話者情報が少なくコンテンツ情報が豊富
- **日本語HuBERT**: アクセントラベル予測89.8%、高低ラベル予測93.2%を達成
- **WavLM残留話者情報**: WavLMの残留話者情報は9.02%（HuBERTの13.72%と比較して大幅に少ない）。VCにおけるWavLMの優位性を裏付け
- **加重和 vs 単一層**: 歌声変換では、**加重和（weighted sum）が単一層選択を一貫して上回る**ことが確認されている
- **Eta-WavLM（ACL 2025）**: 線形方程式による話者情報除去手法。全ベースラインを凌駕し、コンテンツ表現の純度を大幅に向上
- **Spin V2**: ContentVecの微調整版。ContentVecの1/9の学習コストでより高い発音精度を達成
- **推奨**: 固定層ではなく、**複数層の加重和（weighted sum）** を導入。s3prlツールキットの手法が参考になる

### 2.7 実装変更箇所

```
変更ファイル:
  infer/lib/jit/get_hubert.py      - モデルローダーの抽象化（複数バックエンド対応）
  infer/modules/vc/pipeline.py     - 特徴量抽出の差し替え（L191-199）
  infer/modules/train/extract_feature_print.py - 学習用特徴量抽出（L114-129）
  configs/v2/*.json                - ssl_modelパラメータ追加
```

---

## 3. 日本語音声学的分析と現行モデルの課題

### 3.1 日本語母音のフォルマント構造

日本語は5母音（/a, i, ɯ, e, o/）のみ。英語の13-20母音と比べシンプルだが、英語モデルでは以下の問題が発生する。

**成人男性のフォルマント周波数:**

| 母音 | IPA | F1 (Hz) | F2 (Hz) | 英語モデルの問題 |
|------|-----|---------|---------|----------------|
| ア | /a/ | 750-800 | 1100-1200 | 比較的良好 |
| イ | /i/ | 250-300 | 2200-2400 | 比較的良好 |
| **ウ** | **/ɯ/** | **300-350** | **1100-1500** | **英語 /u/ (F2:870Hz)と混同。非円唇が円唇に** |
| エ | /e/ | 450-500 | 1800-2000 | 比較的良好 |
| オ | /o/ | 500-550 | 750-900 | 比較的良好 |

**最大の問題**: 日本語の「ウ」（/ɯ/、非円唇後舌狭母音）が英語の /u/（円唇）に引きずられる。F2の差は200-600Hzに及ぶ。

### 3.2 日本語特有の子音の問題

| 音素 | 日本語 | 英語モデルの誤変換 |
|------|--------|-------------------|
| /ts/（つ） | 歯茎破擦音 [tsɯ] | 英語にない音。/t/ + /s/ に分解される |
| /ɸ/（ふ） | 両唇摩擦音 | 唇歯摩擦音 /f/ に変換される |
| /ɾ/（ら行） | 歯茎はじき音 | /l/ や /r/ に誤変換 |
| 促音（っ） | 1モーラ分の閉鎖 | 閉鎖区間が短縮・消失 |
| 撥音（ん） | 環境同化する鼻音 | 適切な異音が生成されない |

### 3.3 モーラ拍と英語モデルの不適合

- **日本語**: モーラ拍言語。各モーラ100-150ms等時性
- **英語**: 強勢拍言語。強勢間の時間間隔が等しい
- **影響**: HuBERTの20msフレームで1モーラ=5-7.5フレーム。促音（5-7フレームの閉鎖）を英語モデルが正しく表現できない
- **歌唱での深刻度**: 1音符=1モーラの対応が基本。モーラ等時性が崩壊すると歌詞が聞き取れない

### 3.4 歌唱における日本語音声の特性

- **母音主体の旋律**: CV構造のため、メロディの大部分を母音が担う
- **母音の延伸**: ロングトーンでは母音部分が延伸。5母音しかないため音色一貫性が重要
- **母音無声化の抑制**: 歌唱時は発話時と異なり無声化が起きにくい
- **ビブラートとピッチアクセントの分離**: F0抽出時に区別が必要

### 3.5 一般的な日本語VCアーティファクト

**母音関連**: /ɯ/ と /o/ の混同、/ɯ/ の英語的 /u/ 化、長母音の不安定化
**子音関連**: 促音消失、/ɾ/ の /l/ 化、/ɸ/ の /f/ 化
**韻律関連**: ピッチアクセントの平板化、モーラ等時性の崩壊
**歌唱固有**: ロングトーンでのフォルマント不安定、子音アタックの曖昧化

---

## 4. 歌声変換SOTA技術動向

### 4.1 RVCに直接適用可能な改善（既存研究より）

#### kNN-SVC（ICASSP 2025）— 倍音強化と時間的平滑化

RVCのFAISS検索パイプラインに直接組み込める改善:

1. **加法合成による倍音強化**: WavLM/HuBERT特徴量は倍音情報が不足→加法合成で注入
2. **時間的連結コストによる平滑化**: コサイン類似度に時間的コストを加え、フレーム間不連続性を低減

**効果**: 歌唱のレガート表現の自然さ向上、くすんだ音やリンギングアーティファクトの解消

#### SPA-SVC（Interspeech 2024）— 自己教師ありピッチ拡張

- 6-18半音のランダムピッチシフトを学習中に適用
- 追加データ・パラメータ増加なしで品質向上
- ソース・ターゲット間の音域差が大きい場合の品質劣化を解決

#### VibE-SVC（Interspeech 2025）— ビブラート保存

- 離散ウェーブレット変換（DWT, Daubechies1）でF0を低周波+高周波に分解
- 高周波成分=ビブラートスタイル情報を明示的に保存・転送
- 現在のメディアンフィルタによるビブラート減衰問題を解決

### 4.2 代替アーキテクチャからの知見

#### So-VITS-SVC
- 正規化フローによる潜在空間マッピング→日本語母音の長さ・ピッチアクセント制御に有利
- RVCにない利点: 正規化フローの表現力。検索ベースの安定性と統合する余地あり

#### DDSP-SVC
- DSP帰納バイアス→ビブラート・しゃくり・こぶしを物理的に正確にモデル化
- 浅い拡散モデルによるポストエンハンスメント→RVCの後段に追加可能（実証済み）

#### GPT-SoVITS v4
- 5秒の音声でゼロショットTTS、1分でfew-shot TTS
- 中日英韓対応、48kHz出力
- TTSで追加学習データを生成し、RVC学習に活用可能

#### Seed-VC
- ゼロショット歌声変換。話者類似度0.868（OpenVoice 0.755を大幅超過）
- F0条件付け+性別依存ピッチシフトテーブル
- BigVGAN移行で高音域歌声品質が大幅向上

#### HQ-SVC（AAAI 2026）
- デカップルドコーデックでコンテンツ・話者特徴量を同時抽出
- 80時間未満のデータで単一消費者GPUでトレーニング可能
- ゼロショットSVCのSOTAを大幅超過

### 4.3 歌唱専用データセット・モデル

#### SingNet（2025）
- 3,000時間の多言語歌唱音声データセット（オープンソース）
- Wav2vec2・BigVGAN・NSF-HiFiGANの事前学習チェックポイントを公開
- RVCの事前学習モデル強化に直接利用可能

#### 日本語歌唱データセット

| データセット | 内容 | 用途 |
|-------------|------|------|
| 東北きりたん歌唱DB | J-POP歌唱、豊富なアノテーション | 歌声事前学習 |
| JVS-MuSiC | 多話者歌唱音声 | 多話者歌声事前学習 |
| JSUT-song | 27曲童謡歌唱 | 評価用 |
| NIT-SONG070 | 名工大歌唱DB | 研究用 |
| GTSinger（NeurIPS 2024） | 80.59時間、多言語歌唱技法 | 大規模事前学習 |

---

## 5. F0・ピッチ処理の改善

### 5.1 F0メソッド比較

| メソッド | 精度(RPA) | 速度(RTF) | 歌声適性 | 現状 |
|---------|----------|----------|---------|------|
| **RMVPE** | ~95% | 中速 | **最良**（ポリフォニック対応） | ✅ メインパイプライン統合済み |
| **FCPE** | 96.79% | **0.0062（最速）** | 非常に良い（広音域・アニソン向け） | ✅ **メインパイプライン統合済み** |
| CREPE | ~93% | 低速 | 良い（GPU必須） | ✅ 統合済み |
| harvest | - | 低速 | 普通 | ✅ 統合済み |
| pm | - | 最速 | 限定的 | ✅ 統合済み |

### 5.2 FCPEのメインパイプライン統合 ✅ 実装済み

`torchfcpe>=0.0.4` は pyproject.toml の依存関係に含まれており、メインパイプライン (`pipeline.py`) に統合済み。歌声プリセット「アニソン」のデフォルトF0メソッドとして採用。

### 5.3 歌唱向けF0パラメータ推奨

| パラメータ | 現在値 | 歌唱推奨値 | 理由 |
|-----------|--------|-----------|------|
| `f0_min` | 50 Hz | 65 Hz | 基音2次高調波の誤検出防止。リアルタイム版は65Hz |
| `f0_max` | 1100 Hz | 1100-1400 Hz | ソプラノ裏声・ホイッスル対応 |
| RMVPE `thred` | 0.03 | 0.01-0.03 | 歌声の持続音検出感度向上 |
| FCPE `threshold` | 0.006 | 0.003-0.006 | ブレスが多い曲は低めに |
| `filter_radius` | 3 | 1-2 | **ビブラート保存**。3以上はビブラート減衰 |

### 5.4 日本語歌唱の音域

| 声種 | 地声範囲 | 裏声 | f0推奨 |
|------|---------|------|--------|
| 男性（バリトン-テナー） | A2-G4 (110-392Hz) | ~C5 (523Hz) | min=80, max=600 |
| 男性（高音域） | C3-A4 (131-440Hz) | ~E5 (659Hz) | min=80, max=900 |
| 女性（アルト-メゾ） | G3-D5 (196-587Hz) | ~A5 (880Hz) | min=150, max=900 |
| 女性（ソプラノ） | B3-F5 (247-698Hz) | ~C6 (1047Hz) | min=150, max=1100 |

### 5.5 F0量子化の改善検討

| 方式 | 分解能 | 歌声適性 |
|------|--------|---------|
| 256ビン・メルスケール（現行） | 約28セント/ビン | 十分だが改善余地あり |
| 360ビン・セントスケール（RMVPE出力） | 20セント/ビン | ビブラート微細表現可能 |
| 512ビン・セントスケール | 約14セント/ビン | こぶし・しゃくりの精密再現 |

### 5.6 歌声プリセット ✅ 実装済み

`infer/lib/f0_presets.py` で定義。WebUIから選択可能。

```
J-POP:    rmvpe, f0_min=65,  f0_max=1100, filter_radius=1
演歌:     rmvpe, f0_min=65,  f0_max=900,  filter_radius=0（こぶし保存）
アニソン:  fcpe,  f0_min=80,  f0_max=1200, filter_radius=1
話し声:   rmvpe, f0_min=50,  f0_max=800,  filter_radius=3
```

---

## 6. 損失関数・学習手法の改善

### 6.1 現行の損失構成

```
loss_gen_all = loss_gen(1.0) + loss_fm(2.0) + loss_mel(45.0) + loss_kl(1.0)
```

- `loss_gen`: LS-GANジェネレータ損失
- `loss_fm`: Feature matching損失（L1）
- `loss_mel`: 単一解像度メルスペクトログラムL1損失
- `loss_kl`: KLダイバージェンス損失（固定係数）

### 6.2 推奨改善（段階的導入）

#### Phase 1: 低リスク・高効果（即座に導入可能）

**Multi-Resolution STFT損失の追加:** ✅ 実装済み
```
解像度1: n_fft=512,  hop=128, win=512   → 過渡特性（子音アタック）
解像度2: n_fft=1024, hop=256, win=1024  → バランス
解像度3: n_fft=2048, hop=512, win=2048  → 定常スペクトル（母音フォルマント）
```
- 実装: `infer/lib/train/losses.py` の `MultiResolutionSTFTLoss`
- 係数: `c_mrstft=5.0`（`configs/v2/*.json` で設定）
- `auraloss` ライブラリで実装

**KLダイバージェンスのサイクリカルアニーリング:**（未実装・今後検討）
```python
# 現行: c_kl=1.0（固定）
# 改善案: サイクリカルアニーリングで0→1.0を繰り返す
beta(t) = min(1, (t mod T_cycle) / (T_cycle * 0.5)) * beta_max
```
- KL消失問題（posterior collapse）を防ぎ、潜在空間を有効活用

#### Phase 2: 中リスク・中効果

**学習率スケジュールの改善:**
| 項目 | 現在値 | 推奨値 |
|------|--------|--------|
| スケジューラ | ExponentialLR(gamma=0.999875) | CosineAnnealingWarmRestarts |
| warmup | 0 epochs | 5-10 epochs |
| 初期学習率 | 1e-4 | 5e-5（少量データ時） |
| 最小学習率 | （減衰のみ） | 1e-6 |

**正則化の追加（少量データ向け）:** ✅ p_dropout, Weight Decay 実装済み
| 手法 | 値 | 対象 | 状態 |
|------|--------|------|------|
| p_dropout | 0.1 | TextEncoder, ResidualCouplingBlock | ✅ 実装済み |
| Weight Decay | Generator=0.01, Discriminator=0 | AdamW optimizer | ✅ 実装済み |
| EMA | alpha=0.999 | ジェネレータ推論用 | 未実装 |

**勾配蓄積:**
```
accumulation_steps = 4 → 実効バッチサイズ: 4×4 = 16
```

#### Phase 3: F0・フォルマント品質強化

**F0整合性損失:**
```
L_f0 = |f0_input - f0_extracted(y_hat)|_1
```
- 微分可能F0推定器（FCPE）が必要

**フォルマント保存損失:**
```
L_formant = |A_LPC(x) - A_LPC(y_hat)|_2^2
```
- LPCスペクトル包絡の比較。日本語5母音のフォルマント構造保存

**知覚的損失（HuBERTベース）:**
```
L_perceptual = Σ MSE(HuBERT_layer_l(x), HuBERT_layer_l(y_hat))
```
- 既存のHuBERTモデルを流用可能（勾配停止必須）

#### Phase 4: 高度な改善

- コントラスティブ学習（InfoNCE）によるコンテンツ・話者分離強化
- 音素認識損失（CTC損失 + 日本語ASR）
- MRD + CQTディスクリミネータの追加

### 6.3 統合損失関数（現行 → 最終形）

**現行実装（M1完了後）:**
```
L_total = L_adv + 2.0 * L_fm + 45.0 * L_mel + 1.0 * L_kl + 5.0 * L_MRSTFT  ← ✅ 実装済み
```

**将来の拡張案:**
```
L_total = L_adv
        + 2.0 * L_fm
        + 45.0 * L_mel
        + β(t) * L_kl           ← サイクリカルアニーリング（未実装）
        + 5.0 * L_MRSTFT        ← ✅ 実装済み (c_mrstft=5.0)
        + λ_f0 * L_f0           ← 未実装: 1.0-10.0
        + λ_perc * L_perceptual ← 未実装: 0.1-1.0
```

---

## 7. ボコーダ・デコーダの改善

### 7.1 現行アーキテクチャ

GeneratorNSF = HiFi-GAN V1 + Neural Source Filter (hn-NSF)
- アップサンプリング: [12, 10, 2, 2]（積=480=hop_length）
- 活性化: LeakyReLU（slope=0.1）
- ResBlock1: dilated convolution [[1,3,5],[1,3,5],[1,3,5]]

### 7.2 即座に適用可能な改善

#### 1. SnakeBeta活性化関数（BigVGAN方式）
```
SnakeBeta(x) = x + (1/a) * sin²(ax)
```
- LeakyReLUからの置換のみ。周期的誘導バイアスで歌声の高調波再現が大幅向上
- BigVGAN v2で実証済み。Seed-VCでも移行により高音域品質が大幅改善

#### 2. アンチエイリアスフィルタ（Kaiser窓sinc LPF）
- TransposedConvの前後に挿入
- 高音域のピッチアーティファクト削減
- 実装コスト低

#### 3. mel_fminの変更
- 現行: 0.0 Hz → 推奨: 40.0 Hz
- DC成分除去。OpenVPIの歌声ボコーダーで実証済み構成

### 7.3 中期的な改善候補

#### BigVGAN v2（最有力）
- AMP（Anti-Aliased Multi-Periodicity）モジュール
- Multi-Scale Sub-Band CQTディスクリミネータ
- 前バージョンの3倍高速、100倍データで学習
- 44kHz 128band 512xモデルが公開済み

#### Vocos（効率重視）
- ConvNeXtバックボーン + iSTFTヘッド
- HiFi-GANの13倍高速、BigVGANの70倍高速
- 等方性アーキテクチャ（アップサンプリング不要）
- 歌声の高調波復元で周期性アーティファクトが少ない

#### HiFTNet（歌声最適）
- iSTFTNet + Harmonic-plus-Noise Source Filter
- BigVGAN同等品質で4倍高速、パラメータ1/6
- F0推定ネットワーク統合で位相歪み軽減

### 7.4 ディスクリミネータの改善

| 現行 | 追加推奨 |
|------|---------|
| MPD（周期: 2,3,5,7,11,17,23,37） | MRD（Multi-Resolution Discriminator） |
| MSD（DiscriminatorS x1） | CQTディスクリミネータ（ハーモニック構造） |

学習時のみの変更であり推論時の速度影響なし。

### 7.5 推奨メルスペクトログラム構成（日本語歌声48kHz）

| パラメータ | 現在値 | 推奨値 | 理由 |
|-----------|--------|--------|------|
| n_fft | 2048 | 2048 | 適切 |
| hop_length | 480 | 480 | 10ms@48kHz、適切 |
| n_mels | 128 | 128 | 適切 |
| fmin | 0.0 | **40.0** | DC除去、歌声最低音カバー |
| fmax | null | 16000 or null | OpenVPI実証済み |

---

## 8. データ拡張・前処理

### 8.1 歌唱向け前処理パラメータ ✅ 主要項目は実装済み

| パラメータ | 旧値 | 現在値 | 状態 | 理由 |
|-----------|------|--------|------|------|
| Slicer threshold | -42 dB | **-38 dB** | ✅ 実装済み | 歌唱は発声が大きい |
| min_length | 1500 ms | **2000 ms** | ✅ 実装済み | ロングノート対応 |
| per（セグメント長） | 3.7 s | **5.0 s** | ✅ 実装済み | フレーズ長拡張 |
| overlap | 0.3 s | **0.5 s** | ✅ 実装済み | 境界品質向上 |
| ハイパスフィルタ Wn | 48 Hz | **40 Hz** | ✅ 実装済み | 低音域歌唱保護 |
| min_interval | 400 ms | 300 ms | 未実装 | ブレス区間で分割 |
| hop_size | 15 ms | 10 ms | 未実装 | 細かい分解能 |
| max_sil_kept | 500 ms | 300 ms | 未実装 | 歌唱ブレスは短い |

### 8.2 データ拡張戦略（10分データ→40-100分相当）

#### ピッチシフト（最重要、4倍化）
```python
for n_steps in [-4, -2, 2, 4]:  # 半音単位
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
```
- 安全範囲: -6〜+6半音（フォルマント保持）
- 8半音以上はロボット的アーティファクト発生

#### タイムストレッチ（2倍化）
```python
for rate in [0.9, 0.95, 1.05, 1.1]:
    y_stretched = librosa.effects.time_stretch(y, rate=rate)
```

#### SpecAugment（過学習20%削減）
```
freq_mask_param: 15-30（メルビン数の10-20%）
time_mask_param: 50-100（フレーム数の5-10%）
適用率: 50%のバッチに適用
```

#### TTS拡張（GPT-SoVITS活用、3-5倍化）
1. 10分データでRVC/GPT-SoVITSモデルを初期学習
2. 大量の日本語テキストからTTSで合成音声生成
3. 初期RVCでターゲット声質に変換
4. UTMOS > 3.5のサンプルのみ採用
5. 混合比率: 元:合成 = 1:2-3

### 8.3 UVR5ボーカル分離の推奨パイプライン

```
楽曲ファイル
  ↓ [Kim Vocal 2] ボーカル抽出（一次分離）
  ↓ [VR Architecture de-reverb] リバーブ除去
  ↓ [VR Architecture de-echo] エコー除去
  ↓ 手動チェック（アーティファクト箇所除外）
  ↓ SNR > 20dB フィルタリング
  ↓ クリーンボーカル → RVC学習データ
```

### 8.4 転移学習の段階的フロー

```
Step 1: ReazonSpeech（大規模日本語）で一次事前学習
  ↓
Step 2: JVS（100話者マルチスタイル）で二次学習
  ↓
Step 3: 東北きりたん歌唱DB + JVS-MuSiC で歌唱適応
  ↓
Step 4: ターゲット声優10分データでファインチューニング
  - 学習率: 5e-5→2e-5
  - 初期50エポック: エンコーダ凍結、デコーダのみ学習
  - 以降: 全層解凍
```

### 8.5 少量データ向けカリキュラム学習

| Phase | エポック | データ | 学習率 | 特記 |
|-------|---------|-------|--------|------|
| 1: クリーンデータ | 1-100 | SNR上位50%のみ | 1e-4 | 基本声質獲得 |
| 2: 全データ | 101-300 | 全セグメント | 5e-5 | SpecAugment有効化 |
| 3: 難しいデータ重点 | 301-500 | 損失大セグメント重点 | 2e-5 | 弱点補強 |

---

## 9. RVCコミュニティの知見

### 9.1 日本語改善の実績あるプロジェクト

#### ddPn08/rvc-webui
- 日本語HuBERT(`hubert-base-japanese`)を標準サポート
- マルチスピーカー学習対応（話者非依存の発話内容学習を促進）
- nadare氏の事前学習にも使用された基盤

#### nadare氏のRVC v3改造版
- **Causal Convolution**: 未来情報を参照しないリアルタイム特化
- **LoRA統合**: Speaker Embeddingから2行列生成、話者固有の微調整
- **segment_size拡大**: 0.24秒→**1.5秒**（日本語の発話パターンに最適化）
- **bfloat16**: fp16→bf16で学習安定性向上
- **MiniBatchKmeans**: FAISSインデックスの軽量化

#### VoRAS (nadare氏)
- ボコーダをVocosに置換した実験プロジェクト
- 高い女性声でVocosの限界に直面→現在は中断

### 9.2 日本語RVCコミュニティの推奨設定

#### 学習パラメータ

| パラメータ | 推奨値 | 備考 |
|-----------|--------|------|
| エポック数 | データ(分) × エポック = 500-600 | 10分なら50-60エポック |
| バッチサイズ | 30分以上:8、30分未満:4 | VRAM許す限り大きく |
| サンプリングレート | 48kHz | v2標準 |
| F0メソッド | rmvpe | 歌声で最高品質 |
| index_rate | jp-HuBERT時: 0.8-1.0 | 話者情報補正 |

#### データ準備のコツ
- BGM・効果音・ノイズのないクリーン音声
- 様々なシーン（喋り・感情・アカペラ）を含める
- マルチバイト文字（日本語）のファイルパスを避ける

### 9.3 参考リソース

| リソース | URL |
|---------|-----|
| Qiita: RVCの日本語事前学習 | https://qiita.com/nadare/items/18cd74e51c731904c3b0 |
| Qiita: 前編 | https://qiita.com/nadare/items/306521c6010bf3efb115 |
| Zenn: RVC v3軽量化 | https://zenn.dev/aivoicelab/articles/c06de10a4f3f48 |
| BOOTH: jp-HuBERT事前学習済み | https://booth.pm/ja/items/4802383 |
| Kaggle: チェックポイント | https://www.kaggle.com/datasets/nadare/rvc-webui-tuned-weights |
| RVC Wiki（日本語） | https://seesaawiki.jp/rvc_ch/ |
| RVC構造メモ | https://zenn.dev/aivoicelab/articles/f0cd8e735236c6 |
| RVCのしくみ・コツ | https://zenn.dev/mossan_hoshi/articles/20230519_rvc |
| 産総研プレスリリース | https://www.aist.go.jp/aist_j/press_release/pr2025/pr20250310/pr20250310.html |
| imprt/kushinada-hubert-base | https://huggingface.co/imprt/kushinada-hubert-base |
| Spin V2 | https://huggingface.co/IAHispano/Applio/tree/main/Resources/embedders |
| Eta-WavLM | https://arxiv.org/abs/2505.19273 |

---

## 10. 評価手法

### 10.1 客観評価指標

| 指標 | PASS | WARN | FAIL | 測定ツール | 状態 |
|------|------|------|------|-----------|------|
| **MCD** | < 6.0 dB | 6-8 dB | > 8 dB | MCD-DTW (n_mels=40, MFCC 1-12, radius=20) | ✅ 実装済み |
| **F0 RMSE** | < 20 cents | 20-50 | > 50 | RMVPE→harvest fallback, DTW radius=20 | ✅ 実装済み |
| **CER（文字誤り率）** | < 10% | 10-20% | > 20% | **Whisper large-v3**, delta_CER対応 | ✅ 実装済み |
| **話者類似度** | > 0.80 | 0.65-0.80 | < 0.65 | ECAPA-TDNN | 未実装 |
| **UTMOS** | > 4.0 | 3-4 | < 3 | SaruLab/UTMOS22 | 未実装（参考値） |
| **PESQ** | > 3.5 | - | - | `pip install pesq` | 未実装（参考値） |
| **FAD** | < 5.0 | - | - | frechet-audio-distance | 未実装（バッチ評価向き） |

評価CLI: `tools/eval/run_eval.py` (MCD, F0 RMSE, Whisper CER を統合実行)

### 10.2 日本語歌声の最重要評価: Whisper CER ✅ 実装済み

```bash
# 評価CLI経由で実行
uv run python tools/eval/run_eval.py --ref ref.wav --conv conv.wav --metrics cer --whisper-model large-v3
```

- 変換音声をWhisper large-v3で文字起こしし、正解歌詞とのCERを計算
- **delta_CER対応**: 原音CERも測定し、VCによる劣化分のみを評価
- 実装: `tools/eval/metrics/whisper_cer.py`

### 10.3 日本語母音フォルマント分析

Parselmouth（praat-parselmouth、プロジェクト依存に含まれる）で母音のF1/F2を測定。

**判定基準**: 原音と変換音声のF1/F2差が各100Hz以内なら良好。

### 10.4 歌声固有の評価

| 項目 | ツール | 基準 |
|------|--------|------|
| ビブラートRate | Parselmouth | 5-7Hz |
| ビブラートExtent | F0ピーク検出 | 50-200 cents |
| ピッチ精度 | RMVPE | ノート中央値とのcent差 |
| タイミング精度 | mir_eval | オンセット誤差 |

### 10.5 主観評価（MOS）

**推奨フレームワーク**: webMUSHRA（https://github.com/audiolabs/webMUSHRA）

| テスト | 目標 |
|--------|------|
| MOS-N（自然さ） | > 4.0 |
| MOS-S（話者類似度） | > 4.0 |
| MOS-I（日本語明瞭度） | > 4.0 |
| ABX（話者弁別） | 正答率 > 80% |

**設計**: 評価者20名以上、各条件10-20サンプル、5-15秒セグメント

### 10.6 自動評価パイプライン ✅ Stage 1-3 コア部分 実装済み

```
[Stage 1: 信号レベル] MCD + F0 RMSE           ← ✅ 実装済み (tools/eval/metrics/)
  ↓
[Stage 2: 知覚レベル] UTMOS + FAD + フォルマント分析    ← 未実装
  ↓
[Stage 3: 内容レベル] Whisper CER + 話者類似度  ← ✅ CER実装済み / 話者類似度は未実装
  ↓
[Stage 4: 歌声固有] ビブラート + ピッチ精度 + タイミング  ← 未実装
  ↓
[出力] JSONレポート                             ← ✅ 実装済み
```

---

## 11. 統合改善ロードマップ

### Phase 1: 基盤整備（1-2週間）— M0/M1として実装完了

**目標**: 最小の変更で最大の日本語品質改善を達成

| 施策 | 変更箇所 | 効果 | 状態 |
|------|---------|------|------|
| rinna/japanese-hubert-base導入 | `get_hubert.py`, `pipeline.py`, `extract_feature_print.py` | 日本語明瞭度の大幅向上 | 未実装 |
| nadare氏の事前学習チェックポイント適用 | `assets/pretrained_v2/` | 日本語対応のベースモデル | 未実装 |
| FCPEのメインパイプライン統合 | `pipeline.py` | ピッチ精度向上+高速化 | ✅ 実装済み |
| 歌唱用前処理パラメータ | `preprocess.py` | セグメント品質向上 | ✅ 実装済み |
| 評価パイプライン構築 | `tools/eval/` | 改善効果の定量測定 | ✅ 実装済み |

### Phase 2: 学習品質向上（2-3週間）— M1として一部実装完了

**目標**: 損失関数と学習手法の改善

| 施策 | 変更箇所 | 効果 | 状態 |
|------|---------|------|------|
| Multi-Resolution STFT損失追加 | `losses.py`, `train.py` | 過渡特性+定常構造の改善 | ✅ 実装済み (c_mrstft=5.0) |
| KLサイクリカルアニーリング | `train.py` | 潜在空間の有効活用 | 未実装 |
| 学習率スケジュール改善 | `train.py` | 学習安定性向上 | 未実装 |
| 正則化追加 (dropout, weight decay) | `models.py`, `train.py` | 過学習防止 | ✅ 実装済み |
| bf16混合精度対応 | `train.py` | 学習安定性向上 | ✅ 実装済み |
| segment_size拡張 | `configs/v2/*.json` | 歌声フレーズ文脈学習 | ✅ 実装済み (48k=34560, 32k=25600) |
| 歌声プリセット | `infer/lib/f0_presets.py` | ジャンル別最適化 | ✅ 実装済み |
| データ拡張パイプライン | 新規モジュール | 実効データ量4-5倍 | 未実装 |

### Phase 3: ボコーダ改善（3-4週間）

**目標**: 生成音声品質の根本的向上

| 施策 | 変更箇所 | 効果 |
|------|---------|------|
| SnakeBeta活性化関数 | `models.py` (GeneratorNSF) | 歌声高調波再現向上 |
| アンチエイリアスフィルタ | `models.py` | 高音域アーティファクト削減 |
| MRD/CQTディスクリミネータ | `models.py`, `train.py` | ハーモニック構造学習改善 |
| セグメント長拡張 (17280→34560+) | `configs/v2/*.json` | 歌声フレーズ文脈学習 |

### Phase 4: 高度な最適化（4-6週間）

**目標**: SOTA品質への到達

| 施策 | 変更箇所 | 効果 |
|------|---------|------|
| 層加重和（weighted sum） | `pipeline.py`, `get_hubert.py` | SSL特徴量の最適化 |
| F0整合性損失 | `losses.py`, `train.py` | ピッチ精度の明示的強制 |
| ビブラート保存処理（DWT） | `pipeline.py` | 歌唱表現の保存 |
| kNN-SVC加法合成 | `pipeline.py` | 倍音強化 |
| 浅い拡散モデル後段追加 | 新規モジュール | 最終品質ポリッシュ |

---

## 参考文献・情報源

### Qiita記事・日本語リソース
- [続・RVCのモデルを日本語向けに事前学習する](https://qiita.com/nadare/items/18cd74e51c731904c3b0)
- [RVCのモデルを日本語向けに事前学習する（前編）](https://qiita.com/nadare/items/306521c6010bf3efb115)
- [RVCを軽量化したv3を作ってみた](https://zenn.dev/aivoicelab/articles/c06de10a4f3f48)
- [RVCの構造についてのメモ](https://zenn.dev/aivoicelab/articles/f0cd8e735236c6)

### 論文
- RMVPE: A Robust Model for Vocal Pitch Estimation (arXiv:2306.15412)
- FCPE: A Fast Context-based Pitch Estimation (arXiv:2509.15140)
- VibE-SVC: Vibrato Extraction for SVC (Interspeech 2025)
- SPA-SVC: Self-supervised Pitch Augmentation (Interspeech 2024)
- kNN-SVC: k-Nearest Neighbor SVC (ICASSP 2025, arXiv:2504.05686)
- HQ-SVC: High-Quality Zero-Shot SVC (AAAI 2026, arXiv:2511.08496)
- BigVGAN v2 (NVIDIA, 2024)
- Vocos (ICLR 2024, arXiv:2306.00814)
- MERT: Music Understanding (ICLR 2024, arXiv:2306.00107)
- DiffSinger (AAAI 2022)
- Transinger: IPA-based Multilingual Singing (2025)
- YingMusic-SVC: Flow-GRPO for SVC (arXiv:2512.04793)
- SingNet: 3000h Singing Dataset (arXiv:2505.09325)
- GTSinger: NeurIPS 2024 Spotlight

### モデル・データセット
- rinna/japanese-hubert-base: https://huggingface.co/rinna/japanese-hubert-base
- BigVGAN v2: https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x
- OpenVPI SingingVocoders: https://github.com/openvpi/SingingVocoders
- Seed-VC: https://github.com/Plachtaa/seed-vc
- GPT-SoVITS: https://github.com/RVC-Boss/GPT-SoVITS
- Applio: https://github.com/IAHispano/Applio
- ddPn08/rvc-webui: https://github.com/ddPn08/rvc-webui

### 事前学習済みチェックポイント
- BOOTH v2用: https://booth.pm/ja/items/4802383
- Kaggle: https://www.kaggle.com/datasets/nadare/rvc-webui-tuned-weights
