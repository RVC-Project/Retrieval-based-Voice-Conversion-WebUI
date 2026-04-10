# SSLモデル追加調査レポート — 日本語歌声変換の最適な特徴量抽出器

> 5エージェント並列調査（2026-04-10）

---

## 結論: 推奨モデルランキング（更新版）

前回調査で最有力とした `rinna/japanese-hubert-base` を超える候補が複数発見された。

### RVC直接置換可能（768次元・HuBERT Base互換）

| 順位 | モデル | 学習データ | 組織 | ライセンス | HuggingFace |
|------|--------|-----------|------|-----------|-------------|
| **1** | **imprt/kushinada-hubert-base** | **62,215h 日本語** | 産総研 | Apache 2.0 | [Link](https://huggingface.co/imprt/kushinada-hubert-base) |
| 2 | reazon-research/japanese-wav2vec2-base | 35,000h+ 日本語 | Reazon | Apache 2.0 | [Link](https://huggingface.co/reazon-research/japanese-wav2vec2-base) |
| 3 | rinna/japanese-hubert-base | 19,000h 日本語 | rinna | Apache 2.0 | [Link](https://huggingface.co/rinna/japanese-hubert-base) |
| 4 | rinna/japanese-data2vec-audio-base | 19,000h 日本語 | rinna | Apache 2.0 | [Link](https://huggingface.co/rinna/japanese-data2vec-audio-base) |
| 5 | rinna/japanese-wav2vec2-base | 19,000h 日本語 | rinna | Apache 2.0 | [Link](https://huggingface.co/rinna/japanese-wav2vec2-base) |
| 6 | sarulab-speech/hubert-base-jtube | 2,720h 日本語 | 猿渡研(東大) | MIT | [Link](https://huggingface.co/sarulab-speech/hubert-base-jtube) |
| 7 | mHuBERT-147 | 90,000h / 147言語 | UTTER/Naver | Open | [Link](https://huggingface.co/utter-project/mHuBERT-147) |

### VC品質で評価された非日本語特化モデル

| 順位 | モデル | 特徴 | VC実績 |
|------|--------|------|--------|
| 1 | **WavLM-Large (layer 6)** | 残留話者情報**最少9.02%** | kNN-VC, LinearVC, Eta-WavLM採用 |
| 2 | **Spin V2** | ContentVecより発音精度↑、学習コスト1/9 | Applio, RingFormer採用 |
| 3 | ContentVec (現行) | 話者情報除去に特化 | RVC/So-VITS標準 |

### 歌声・音楽特化SSL

| モデル | 特徴 | VISinger2+でのMOS |
|--------|------|-------------------|
| **MERT v1** | 音楽理解特化、ビート・ピッチ・音色 | **3.72（最高）** |
| **MuQ** (Tencent, 2025) | 0.9K時間で既存音楽SSL超え | - |
| SingNet wav2vec2 | 3,000h歌声で事前学習 | - |

---

## 最有力候補: kushinada-hubert-base（産総研）

### なぜkushinadaが最有力か

| 項目 | kushinada | rinna | ContentVec(現行) |
|------|-----------|-------|-----------------|
| 学習データ量 | **62,215h** | 19,000h | 960h |
| 言語 | **日本語100%** | 日本語100% | 英語100% |
| アーキテクチャ | HuBERT Base | HuBERT Base | HuBERT Base |
| 出力次元 | **768（RVC互換）** | 768 | 768 |
| レイヤー数 | 12 | 12 | 12 |
| ライセンス | **Apache 2.0** | Apache 2.0 | MIT |
| データソース | TV放送（多様な話者・ジャンル） | TV放送 | 英語読み上げ |
| 公開日 | **2025年3月** | 2023年4月 | 2022年 |
| ベンチマーク(SER) | **84.77%** | - | - |
| HF DL数(Large版) | 10,800+ | - | - |

**kushinadaはrinnaの3.3倍のデータで学習**されており、テレビ放送由来のため話者多様性・ジャンル多様性が高い。768次元・12層のHuBERT Baseアーキテクチャで、RVCのContentVecとドロップイン置換可能。

---

## 日本語768次元SSLモデル 全カタログ

### 事前学習モデル（特徴量抽出用）

| # | モデル | 組織 | データ | 時間 | アーキテクチャ | 公開日 | ライセンス |
|---|--------|------|--------|------|---------------|--------|-----------|
| 1 | imprt/kushinada-hubert-base | 産総研 | TV放送 | 62,215h | HuBERT Base | 2025/03 | Apache 2.0 |
| 2 | reazon-research/japanese-wav2vec2-base | Reazon | RS v2.0 | 35,000h+ | wav2vec2 Base | 2024/10 | Apache 2.0 |
| 3 | rinna/japanese-hubert-base | rinna | RS v1 | 19,000h | HuBERT Base | 2023/04 | Apache 2.0 |
| 4 | rinna/japanese-wav2vec2-base | rinna | RS v1 | 19,000h | wav2vec2 Base | 2024/03 | Apache 2.0 |
| 5 | rinna/japanese-data2vec-audio-base | rinna | RS v1 | 19,000h | data2vec Base | 2024/03 | Apache 2.0 |
| 6 | sarulab-speech/hubert-base-jtube | 猿渡研 | YouTube | 2,720h | HuBERT Base | - | MIT |
| 7 | mHuBERT-147 | UTTER | 多言語 | 90,000h | HuBERT Base | 2024/06 | Open |

### 1024次元モデル（要次元調整）

| # | モデル | 組織 | データ | 時間 | アーキテクチャ | ライセンス |
|---|--------|------|--------|------|---------------|-----------|
| 8 | imprt/kushinada-hubert-large | 産総研 | TV放送 | 62,215h | HuBERT Large | Apache 2.0 |
| 9 | rinna/japanese-hubert-large | rinna | RS v1 | 19,000h | HuBERT Large | Apache 2.0 |
| 10 | reazon-research/japanese-wav2vec2-large | Reazon | RS v2.0 | 35,000h+ | wav2vec2 Large | Apache 2.0 |
| 11 | WavLM-Large | Microsoft | 多言語 | 94,000h | WavLM Large | MIT |
| 12 | w2v-BERT 2.0 | Meta | 多言語 | 4,500,000h | Conformer | MIT |

---

## VC品質に関するベンチマーク知見

### SSL層ごとの情報分布

| 層 | エンコードする主な情報 | VC推奨 |
|----|---------------------|--------|
| 低層(1-6) | 話者ティンバー、プロソディ、音響パターン | kNN-VC: **layer 6**を推奨 |
| 中層(7-12) | 話者IDと音韻の混合 | RVC: layer 12を使用 |
| 高層(13-24) | 言語コンテンツ、セマンティクス | ASR向き |

### 残留話者情報の比較（少ないほどVC向き）

| モデル | 残留話者情報 |
|--------|------------|
| **WavLM Base+** | **9.02%** |
| HuBERT Base | 13.72% |
| HuBERT Large L18 | 10.58% |
| ContentVec | HuBERT比36%削減 |

### Soft Units vs Discrete Units

**Soft Unitsが優位**。離散ユニットより明瞭性・自然性が高く、破擦音/摩擦音/軟口蓋閉鎖音の再現で特に優れる。

### 重み付き和（Weighted Sum）の効果

単一層出力より**一貫して優れる**。歌声変換ではメロディ情報抽出を大幅に改善。

---

## 新発見モデルの詳細

### Spin V2（話者不変SSL微調整）

- **原理**: 話者摂動（フォルマント+F0ランダムスケーリング+ランダムEQ）による不変学習
- **性能**: 単一GPUで45分の微調整でHuBERT/ContentVecを上回る発音精度
- **RVC実績**: Applio標準搭載。RingFormer（RVCフォーク）でも採用
- **入手先**: [Applio Embedders](https://huggingface.co/IAHispano/Applio/tree/main/Resources/embedders)

### DSFF-SVC（複数SSL融合、ICASSP 2025）

WeNet + Whisper + ContentVecの3モデルを段階的融合:
- 融合でCER: 38.2% → 15.8%（韻律特徴追加時）
- **複数SSLの融合が単体より大幅に品質向上**

### Eta-WavLM（ACL 2025）

- 線形方程式 `s = f(d) + η` で話者成分を除去
- 全ベースラインを上回るMOS、LJSpeechでほぼground truthに近いWER/PER

### MuQ（Tencent, 2025年1月）

- Mel-RVQ自己教師あり音楽表現学習
- **0.9K時間のオープンデータで既存音楽SSL超え**
- GitHub: [tencent-ailab/MuQ](https://github.com/tencent-ailab/MuQ)

---

## 実装推奨

### 短期（即座に着手可能）

```
第1候補: imprt/kushinada-hubert-base
  - 62,215h日本語、768次元、HuBERT Base、Apache 2.0
  - RVCのContentVecとドロップイン置換可能
  - 変更: get_hubert.py, pipeline.py, extract_feature_print.py

第2候補: rinna/japanese-hubert-base
  - 19,000h日本語、768次元、nadare氏の実績あり
  - BOOTH公開の事前学習済みチェックポイントが利用可能
```

### 中期（比較実験）

```
以下のモデルでA/B比較実験を実施:
  1. kushinada-hubert-base（62,215h日本語）
  2. rinna/japanese-hubert-base（19,000h日本語、既存実績）
  3. Spin V2（話者不変微調整、発音精度最高）
  4. ContentVec（現行ベースライン）

評価指標: Whisper CER + MCD + 話者類似度 + 主観MOS
```

### 長期（アーキテクチャ拡張）

```
検討事項:
  - 重み付き和（Weighted Sum of Layers）の導入
  - DSFF-SVC式の複数SSL融合（kushinada + Whisper）
  - WavLM-Large layer 6 の評価（1024→768射影必要）
  - MERT統合の歌声特化実験
```

---

## 参考リンク

### 日本語SSLモデル
- [産総研プレスリリース: いざなみ・くしなだ](https://www.aist.go.jp/aist_j/press_release/pr2025/pr20250310/pr20250310.html)
- [rinna 日本語音声モデル公開(2024年3月)](https://rinna.co.jp/news/2024/03/20240307.html)
- [ReazonSpeech wav2vec2公開(2024年10月)](https://research.reazon.jp/blog/2024-10-21-Wav2Vec2-base-release.html)

### VC用SSL比較論文
- [kNN-VC (Interspeech 2023)](https://github.com/bshall/knn-vc) — WavLM layer 6
- [LinearVC (Interspeech 2025)](https://arxiv.org/html/2506.01510) — WavLM layer 6
- [ContentVec (ICML 2022)](https://arxiv.org/abs/2204.09224) — 話者情報36%削減
- [Eta-WavLM (ACL 2025)](https://arxiv.org/abs/2505.19273) — 線形話者除去
- [DSFF-SVC (ICASSP 2025)](https://arxiv.org/abs/2310.11160) — 複数SSL融合
- [Spin (Interspeech 2023)](https://arxiv.org/pdf/2305.11072) — 話者不変学習

### 歌声・音楽SSL
- [MERT (ICLR 2024)](https://github.com/yizhilll/MERT)
- [MuQ (Tencent 2025)](https://github.com/tencent-ailab/MuQ)
- [SingNet (2025)](https://arxiv.org/abs/2505.09325)

### RVCコミュニティ
- [Applio Embedder一覧](https://huggingface.co/IAHispano/Applio/tree/main/Resources/embedders)
- [RingFormer事前学習](https://huggingface.co/its5Q/rvc-ringformer-pretrain)
- [nadare氏のRVC日本語事前学習(Qiita)](https://qiita.com/nadare/items/18cd74e51c731904c3b0)
