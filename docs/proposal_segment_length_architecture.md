# 提案書: セグメント長拡張とモデルアーキテクチャ調整

## 概要

本提案は、RVC WebUIにおける日本語歌唱音声変換品質の改善を目的として、トレーニング時のセグメント長の拡張およびモデルアーキテクチャの調整を行うものである。

現在の48kHz設定では`segment_size=17280`（約0.36秒）でトレーニングセグメントが切り出されるが、日本語の1モーラは約100-150msであるため、0.36秒では2-3モーラ分しかカバーできず、歌唱フレーズの文脈が著しく断片化される。nadare氏のRVC v3改造版では1.5秒への拡張により品質が大幅に向上したことが実証されており、本プロジェクトでも同様のアプローチを段階的に適用する。

併せて、TextEncoderのヘッド数・層数の増加、bfloat16への移行、LoRA統合、Causal Convolutionの導入を提案する。

### 対象ファイル

| ファイル | 変更内容 |
|---------|---------|
| `configs/v2/48k.json` | segment_size、n_heads、n_layers等の設定変更 |
| `configs/v2/32k.json` | 同上（32kHz版） |
| `infer/lib/infer_pack/models.py` | SynthesizerTrnMs768NSFsid、TextEncoder、ResidualCouplingBlockの改修 |
| `infer/lib/infer_pack/attentions.py` | Encoder/MultiHeadAttentionの改修 |
| `infer/lib/infer_pack/modules.py` | Causal Convolution対応、LoRAレイヤ追加 |
| `infer/modules/train/train.py` | bfloat16対応、gradient checkpointing導入 |
| `infer/lib/infer_pack/commons.py` | slice_segments関数の拡張セグメント対応 |

---

## 技術的詳細

### 1. セグメント長拡張

#### 現行パラメータ（48kHz）

```
segment_size = 17280 samples
hop_length = 480
segment_frames = 17280 / 480 = 36 frames
segment_duration = 17280 / 48000 = 0.36 seconds
```

#### 提案する3段階のセグメント長

| 段階 | segment_size | 秒数 | フレーム数 | カバーモーラ数 |
|------|-------------|------|-----------|-------------|
| 現行 | 17,280 | 0.36s | 36 | 2-3 |
| Step 1 | 34,560 | 0.72s | 72 | 5-7 |
| Step 2 | 48,000 | 1.00s | 100 | 7-10 |
| Step 3 | 72,000 | 1.50s | 150 | 10-15 |

#### VRAM使用量の詳細計算

VRAMの主要な消費源は以下の4つである。

**(A) Generator（GeneratorNSF）のアクティベーション**

GeneratorNSFのupsample_rates = [12, 10, 2, 2]（総アップサンプル倍率 = 480）。
segment_sizeのサンプル数がそのまま波形出力長に対応する。

各アップサンプルステージのアクティベーション:
- Stage 0入力: segment_frames個 x 512ch → 出力: x12
- Stage 1: x10
- Stage 2: x2
- Stage 3: x2

Generatorアクティベーション（fp16、batch_size=4）:

| segment_size | 中間表現合計 | Generator VRAM |
|-------------|------------|---------------|
| 17,280 | ~158M params/activations | ~0.6 GB |
| 34,560 | ~316M | ~1.2 GB |
| 48,000 | ~439M | ~1.7 GB |
| 72,000 | ~658M | ~2.5 GB |

**(B) Discriminator（MultiPeriodDiscriminator）のアクティベーション**

DiscriminatorSおよびDiscriminatorP（periods=[2,3,5,7,11,17]の6つ + DiscriminatorS 1つ = 計7サブネット）が入力波形を処理する。波形長はsegment_sizeに比例するため:

| segment_size | Discriminator VRAM |
|-------------|-------------------|
| 17,280 | ~0.8 GB |
| 34,560 | ~1.6 GB |
| 48,000 | ~2.2 GB |
| 72,000 | ~3.3 GB |

**(C) PosteriorEncoder + Flow + TextEncoderのアクティベーション**

これらはフレーム数（segment_size / hop_length）に比例する。
PosteriorEncoder: WN(192ch, kernel=5, dilation=1, 16層) + Conv1d。
Flow: 4 ResidualCouplingLayer x 3層（kernel=5）。
TextEncoder: Transformer 6層、hidden=192、heads=2。

Attention行列のメモリ: `batch * heads * T * T * 2bytes(fp16)`

| segment_size | frames (T) | Attention行列 | Encoder全体 VRAM |
|-------------|-----------|--------------|----------------|
| 17,280 | 36 | 4 * 2 * 36 * 36 * 2 = 20 KB | ~0.1 GB |
| 34,560 | 72 | 4 * 2 * 72 * 72 * 2 = 82 KB | ~0.2 GB |
| 48,000 | 100 | 4 * 2 * 100 * 100 * 2 = 160 KB | ~0.3 GB |
| 72,000 | 150 | 4 * 2 * 150 * 150 * 2 = 360 KB | ~0.4 GB |

注: TextEncoderのAttentionはwindow_size=10の相対位置エンコーディングを使用しているため、実際にはsparse attentionに近く、T^2の全結合よりは小さい。ただしFFN層（filter_channels=768）のアクティベーションはTに比例して増加する。

**(D) 総VRAM推定（batch_size=4、fp16）**

| segment_size | 秒数 | Generator | Discriminator | Enc/Flow | パラメータ+Optimizer | **合計** |
|-------------|------|-----------|---------------|----------|-------------------|---------|
| 17,280 | 0.36s | 0.6 GB | 0.8 GB | 0.1 GB | 1.5 GB | **~3.0 GB** |
| 34,560 | 0.72s | 1.2 GB | 1.6 GB | 0.2 GB | 1.5 GB | **~4.5 GB** |
| 48,000 | 1.00s | 1.7 GB | 2.2 GB | 0.3 GB | 1.5 GB | **~5.7 GB** |
| 72,000 | 1.50s | 2.5 GB | 3.3 GB | 0.4 GB | 1.5 GB | **~7.7 GB** |

注: パラメータ+Optimizerの1.5 GBはsegment_sizeに依存せず一定。内訳はモデルパラメータ(fp16)約56M params = ~112MB、Optimizer State(fp32 momentum + variance) = ~450MB x 2(G+D) = ~900MB、その他バッファ約0.5GB。

#### batch_size調整込みの実用VRAM推定

| GPU | VRAM | 推奨segment_size | 推奨batch_size | 実効VRAM |
|-----|------|----------------|---------------|---------|
| RTX 3060 | 12 GB | 34,560 (0.72s) | 4 | ~4.5 GB |
| RTX 3080 | 10 GB | 34,560 (0.72s) | 3 | ~3.8 GB |
| RTX 3090 | 24 GB | 72,000 (1.50s) | 4 | ~7.7 GB |
| RTX 4060 Ti | 16 GB | 48,000 (1.00s) | 4 | ~5.7 GB |
| RTX 4080 | 16 GB | 48,000 (1.00s) | 6 | ~7.5 GB |
| RTX 4090 | 24 GB | 72,000 (1.50s) | 4 | ~7.7 GB |
| A100 | 40/80 GB | 72,000 (1.50s) | 8-16 | ~15-30 GB |

gradient checkpointingを有効化した場合、アクティベーションメモリを約40-60%削減可能（計算時間は20-30%増加）。

---

### 2. TextEncoderの拡張

#### 現行アーキテクチャ

```python
# configs/v2/48k.json
"n_heads": 2,
"n_layers": 6,
"hidden_channels": 192,
"filter_channels": 768,

# attentions.py - MultiHeadAttention
k_channels = channels // n_heads  # 192 // 2 = 96
```

#### 提案するアーキテクチャ

```python
# 提案
"n_heads": 4,       # 2 → 4
"n_layers": 8,      # 6 → 8
"hidden_channels": 192,    # 変更なし
"filter_channels": 768,    # 変更なし

# 結果
k_channels = 192 // 4 = 48  # ヘッド当たりの次元は96→48に縮小
```

#### 根拠

**ヘッド数 2→4:**
- 192次元を2ヘッドで分割すると各ヘッド96次元。これは音声特徴量に対して過剰に大きい
- 4ヘッド（各48次元）にすることで、異なるアテンションパターンを並列学習できる
- 日本語歌唱では母音の長さ、子音の立ち上がり、ピッチ変化、ビブラートなど複数の特徴を同時に捉える必要があり、ヘッド数の増加が有効
- 48次元は音声特徴量のヘッドサイズとして標準的（VITSオリジナルも類似の設定）

**層数 6→8:**
- セグメント長が長くなると、より長い文脈依存性を捉える必要がある
- 8層にすることで受容野が拡大し、0.72-1.5秒のフレーズ全体を考慮した特徴量変換が可能になる
- 計算コスト増加は約33%だが、Encoderは全体の計算時間の小さな割合であるため影響は限定的

#### パラメータ数の変化

| コンポーネント | 現行(h=2,l=6) | 提案(h=4,l=8) | 増加率 |
|-------------|-------------|-------------|-------|
| TextEncoder全体 | ~3.2M | ~4.3M | +34% |
| うちAttention (QKV+O) | ~0.89M | ~1.18M | +33% |
| うちFFN | ~1.77M | ~2.36M | +33% |
| モデル全体 | ~55.7M | ~56.8M | +2.0% |

TextEncoderはモデル全体のパラメータ数のうち約5.7%を占めるに過ぎず、層数・ヘッド数の増加はモデル全体への影響が小さい。

---

### 3. 混合精度: fp16 → bfloat16

#### 現行の実装

```python
# train.py
from torch.amp import GradScaler, autocast
scaler = GradScaler(enabled=hps.train.fp16_run)
with autocast("cuda", enabled=hps.train.fp16_run):
    # fp16で前方伝搬
```

#### 提案する変更

```python
# train.py 変更案
use_bf16 = hps.train.get("bf16_run", False)
use_fp16 = hps.train.fp16_run and not use_bf16

if use_bf16:
    # bfloat16ではGradScalerは不要（ダイナミックレンジがfp32と同等）
    scaler = GradScaler(enabled=False)
    amp_dtype = torch.bfloat16
else:
    scaler = GradScaler(enabled=use_fp16)
    amp_dtype = torch.float16

with autocast("cuda", dtype=amp_dtype, enabled=use_fp16 or use_bf16):
    # 演算実行
```

#### メリット

| 特性 | fp16 | bfloat16 |
|-----|------|---------|
| 指数部ビット | 5 | 8 |
| 仮数部ビット | 10 | 7 |
| ダイナミックレンジ | ~6.5x10^4 | ~3.4x10^38 |
| GradScaler必要性 | 必須 | 不要 |
| 精度 | 高い | やや低い |
| loss spike発生率 | 中〜高 | 低 |

- bfloat16はfp32と同じ指数部ビット数を持ち、オーバーフロー/アンダーフローが起きにくい
- GradScalerが不要になり、学習の安定性が向上
- nadare氏のv3改造でもbfloat16への移行を実施し、効果を確認済み
- 要件: Ampere以降のGPU（RTX 30xx/A100以降）。Turing（RTX 20xx）以前ではハードウェアサポートなし

---

### 4. LoRA統合

#### 設計概要

nadare氏のv3改造に倣い、Speaker Embeddingからrank分解した2つの行列を生成し、pointwise convolution（1x1 Conv）にLoRAアダプタとして適用する。

```python
class LoRAConv1d(nn.Module):
    """Conv1dに対するLoRA（Low-Rank Adaptation）"""
    def __init__(self, conv: nn.Conv1d, rank: int = 4, gin_channels: int = 256):
        super().__init__()
        self.conv = conv
        self.rank = rank
        in_ch = conv.in_channels
        out_ch = conv.out_channels
        # Speaker embeddingからLoRA行列を生成
        self.lora_down_gen = nn.Linear(gin_channels, in_ch * rank)
        self.lora_up_gen = nn.Linear(gin_channels, rank * out_ch)
        self.scale = 1.0 / rank

    def forward(self, x, g=None):
        out = self.conv(x)
        if g is not None:
            g_flat = g.squeeze(-1)  # [B, gin_channels]
            B = x.size(0)
            # LoRA行列を生成
            lora_down = self.lora_down_gen(g_flat).view(B, self.rank, -1)  # [B, rank, in_ch]
            lora_up = self.lora_up_gen(g_flat).view(B, -1, self.rank)     # [B, out_ch, rank]
            # バッチごとにLoRA適用
            delta = torch.bmm(lora_up, torch.bmm(lora_down, x)) * self.scale  # [B, out_ch, T]
            out = out + delta
        return out
```

#### 適用箇所

| モジュール | レイヤ | 適用理由 |
|----------|-------|---------|
| TextEncoder | emb_phone (Linear→Conv1d化) | 話者固有の音素埋め込み調整 |
| TextEncoder | proj (Conv1d) | 話者ごとの出力分布調整 |
| PosteriorEncoder | pre, proj (Conv1d) | 話者固有のスペクトル処理 |
| Flow | ResidualCouplingLayerのpointwise conv | 話者ごとの潜在空間変換 |

#### パラメータ増加

rank=4の場合、1つのLoRAConv1dあたり: `gin_channels * (in_ch * rank + rank * out_ch)`
典型的な192ch conv: `256 * (192*4 + 4*192) = 256 * 1536 = ~393K params`
全適用箇所（約10-15レイヤ）: ~4-6M params追加（モデル全体の約8-10%増）

---

### 5. Causal Convolution

#### 目的

通常の畳み込みは入力の前後の文脈を参照するが、Causal Convolutionは過去のフレームのみを参照する。リアルタイム推論において未来情報に依存しないため、ストリーミング処理が可能になる。

#### 変更箇所

```python
# modules.py - WN（WaveNet）の畳み込み
# 現行: 標準的なdilated convolution（双方向）
padding = int((kernel_size * dilation - dilation) / 2)

# 提案: causal padding（左側のみにパディング）
if causal:
    padding = (kernel_size - 1) * dilation  # 左側のみにフルパディング
else:
    padding = int((kernel_size * dilation - dilation) / 2)
```

#### 影響範囲

- PosteriorEncoder内のWN: 16層のdilated causal conv
- ResidualCouplingLayer内のWN: 3層 x 4フロー
- Generator/Discriminatorは波形領域で動作するため変更不要

---

### 6. 32kHz設定への対応

32kHzの場合、hop_length=320であるため:

| 段階 | segment_size (32k) | 秒数 | フレーム数 |
|------|-------------------|------|-----------|
| 現行 | 12,800 | 0.40s | 40 |
| Step 1 | 25,600 | 0.80s | 80 |
| Step 2 | 32,000 | 1.00s | 100 |
| Step 3 | 48,000 | 1.50s | 150 |

---

## セグメント長 vs VRAM vs 品質 トレードオフ表

| segment_size | 秒数 | 日本語モーラ | VRAM (bs=4, fp16) | VRAM (bs=4, bf16) | VRAM (bs=4, bf16, GC) | 学習速度比 | 期待品質 |
|-------------|------|-----------|------------------|------------------|---------------------|-----------|---------|
| 17,280 | 0.36s | 2-3 | 3.0 GB | 3.0 GB | 2.2 GB | 1.00x (基準) | 基準 |
| 34,560 | 0.72s | 5-7 | 4.5 GB | 4.5 GB | 3.2 GB | 0.75x | +15-20% |
| 48,000 | 1.00s | 7-10 | 5.7 GB | 5.7 GB | 4.0 GB | 0.60x | +25-30% |
| 72,000 | 1.50s | 10-15 | 7.7 GB | 7.7 GB | 5.5 GB | 0.45x | +30-40% |

GC = Gradient Checkpointing有効時（アクティベーションメモリ約40%削減、計算時間約25%増加）

### 品質向上の根拠

- **0.36s（現行）**: 「さくら」の3モーラすらカバーしきれない。子音と母音の遷移が切断される頻度が高い
- **0.72s**: 「おはよう」（4モーラ）程度の語をカバー。短いフレーズの音声的文脈を維持できる
- **1.00s**: 歌唱の1小節（BPM120で2拍）をカバー。ビブラートの1-2周期を含む
- **1.50s**: 歌唱の2小節分。フレーズ単位の韻律をモデルが学習可能。nadare氏が実証済みの長さ

### batch_size削減時の学習効率

segment_sizeを大きくするとbatch_sizeの削減が必要になる場合がある。

| segment_size | batch_size=4 VRAM | batch_size=2 VRAM | batch_size=1 VRAM |
|-------------|------------------|------------------|------------------|
| 34,560 | 4.5 GB | 3.0 GB | 2.3 GB |
| 48,000 | 5.7 GB | 3.8 GB | 2.8 GB |
| 72,000 | 7.7 GB | 5.1 GB | 3.8 GB |

batch_size削減のデメリット:
- 勾配の分散が大きくなり学習が不安定になる可能性
- 1エポックあたりのステップ数が増加（時間増）
- gradient accumulation（勾配蓄積）で仮想バッチサイズを維持可能

---

## メリット

### 1. 日本語歌唱品質の大幅向上
セグメント長の拡大により、日本語のモーラ連鎖や歌唱フレーズの文脈をモデルが学習できるようになる。特に語頭子音の明瞭度、母音間の滑らかな遷移、ビブラートの自然さが改善される。nadare氏の実証により、1.5秒への拡張で品質が顕著に向上することが確認されている。

### 2. 学習安定性の向上（bfloat16）
bfloat16への移行によりGradScalerが不要になり、loss spikeの発生頻度が大幅に低下する。特にセグメント長を拡大した際のメモリ増加と相まって起きやすいオーバーフロー問題を根本的に解決する。

### 3. TextEncoderの表現力向上
4ヘッド化により、ピッチ・タイミング・音色・言語的特徴を独立したヘッドで並列処理でき、特に日本語歌唱で重要な母音の質（「い」と「え」の区別など）の改善が期待される。8層化により、より長い距離の文脈依存性を捉えられる。

### 4. 話者適応の効率化（LoRA）
LoRA統合により、少量の追加パラメータ（全体の8-10%）で話者固有の微調整が可能になる。事前学習モデルの本体を凍結したまま、LoRA部分のみをファインチューニングすることで学習効率が向上する。

### 5. リアルタイム推論の最適化（Causal Convolution）
Causal Convolutionの導入により、ストリーミング処理が可能になる。現行のツール `tools/rvc_for_realtime.py` との親和性が高まり、レイテンシの予測可能性が向上する。

### 6. 段階的導入が可能
セグメント長の拡大は設定ファイルの変更のみで段階的に試行でき、34,560（0.72s）から始めて効果を確認しながら拡大できる。アーキテクチャ変更も独立して適用可能。

### 7. 既存の事前学習モデルとの互換性管理
セグメント長はトレーニング時のスライシングパラメータであり、推論時のモデル構造には影響しない。TextEncoderの変更は事前学習からのやり直しが必要だが、段階的に導入可能。

---

## デメリット・リスク

### 1. VRAM使用量の大幅増加
segment_size=72,000ではVRAM使用量が現行の約2.6倍（3.0GB→7.7GB）に増加する。RTX 3060（12GB）以下のGPUではbatch_sizeの削減やgradient checkpointingが必須となり、学習速度が低下する。

### 2. 事前学習モデルの非互換
TextEncoderのヘッド数・層数を変更した場合、既存の事前学習モデル（`assets/pretrained_v2/`）のウェイトをそのまま使用できない。新しいアーキテクチャで事前学習をやり直す必要があり、大規模な計算リソースと時間が必要。segment_sizeのみの変更であれば既存ウェイトは利用可能。

### 3. 学習速度の低下
セグメント長の拡大はGenerator・Discriminatorの計算量を直接的に増加させる。segment_size=72,000では1ステップあたりの計算時間が約2.2倍に増加し、同一エポック数での学習時間が大幅に延びる。

### 4. bfloat16のハードウェア制約
bfloat16はAmpere以降のGPU（RTX 30xx/A100以降）でのみネイティブサポートされる。RTX 20xxやGTX 16xxのユーザーは従来のfp16を使い続ける必要があり、コードの分岐が増える。

### 5. LoRAの学習困難さ
Speaker Embeddingから動的にLoRA行列を生成する設計は、通常の固定LoRAよりも学習が不安定になりやすい。ランク（rank）の選択やscaleの調整に試行錯誤が必要であり、最適なハイパーパラメータがデータセットに依存する。

### 6. Causal Convolutionによる品質低下リスク
Causal Convolutionは未来情報を参照できないため、双方向畳み込みと比較して理論的に品質が低下する可能性がある。特にオフライン処理（後処理として変換する通常のワークフロー）ではメリットがデメリットを上回るか慎重に検証が必要。

### 7. テスト・検証の複雑化
6つの独立した変更（segment拡大、heads、layers、bf16、LoRA、causal conv）を組み合わせると、検証すべきパラメータの組み合わせが膨大になる。ABテストの設計と定量評価（CER、MCD、話者類似度）のパイプライン整備が前提条件となる。

### 8. コミュニティとの互換性
本家RVCや他のフォーク（ddPn08/rvc-webuiなど）とモデルの互換性が失われる可能性がある。特にTextEncoderの変更は、学習済みモデルの共有・配布時に問題となる。

---

## 必要なハードウェア

### 最小要件（Step 1: segment_size=34,560のみ）

| コンポーネント | 要件 |
|-------------|------|
| GPU | NVIDIA RTX 3060以上（12GB VRAM） |
| GPU世代 | Turing以降（fp16のみならTuring可、bf16ならAmpere以降） |
| RAM | 16 GB以上 |
| ストレージ | SSD 50GB以上（データセット+チェックポイント） |
| VRAM使用量 | ~4.5 GB（batch_size=4） |

### 推奨要件（Step 3: segment_size=72,000 + 全アーキテクチャ変更）

| コンポーネント | 要件 |
|-------------|------|
| GPU | NVIDIA RTX 3090/4080以上（24GB VRAM推奨） |
| GPU世代 | Ampere以降（bf16必須） |
| RAM | 32 GB以上 |
| ストレージ | NVMe SSD 200GB以上 |
| VRAM使用量 | ~8-10 GB（アーキテクチャ変更含む、batch_size=4） |

### 事前学習のやり直しが必要な場合

| コンポーネント | 要件 |
|-------------|------|
| GPU | NVIDIA A100 (40/80GB) x 1-4台 |
| 学習データ | 100時間以上の日本語音声 |
| 学習時間 | A100 x4で約3-7日（データ量と設定に依存） |
| コスト概算 | クラウド(A100 x4): 約$500-1,500 |

### GPU別の推奨構成

| GPU | VRAM | 推奨segment_size | batch_size | bf16 | GC | 総VRAM |
|-----|------|-----------------|-----------|------|-----|--------|
| RTX 3060 | 12 GB | 34,560 | 4 | N/A (fp16) | 不要 | ~4.5 GB |
| RTX 3070 Ti | 8 GB | 34,560 | 2 | N/A (fp16) | 推奨 | ~2.5 GB |
| RTX 3080 | 10 GB | 34,560 | 4 | N/A (fp16) | 不要 | ~4.5 GB |
| RTX 3090 | 24 GB | 72,000 | 4 | 非対応(Ampere bf16は可) | 不要 | ~7.7 GB |
| RTX 4060 Ti | 16 GB | 48,000 | 4 | 可 | 不要 | ~5.7 GB |
| RTX 4070 | 12 GB | 34,560 | 4 | 可 | 不要 | ~4.5 GB |
| RTX 4080 | 16 GB | 48,000 | 6 | 可 | 不要 | ~7.5 GB |
| RTX 4090 | 24 GB | 72,000 | 6 | 可 | 不要 | ~11 GB |
| A100 40GB | 40 GB | 72,000 | 12 | 可 | 不要 | ~22 GB |

---

## 実装工数

### Phase 1: セグメント長拡張のみ（最小変更）

| タスク | 工数 | 変更ファイル |
|-------|------|------------|
| configs/v2/48k.json のsegment_size変更 | 0.5h | 1ファイル |
| configs/v2/32k.json のsegment_size変更 | 0.5h | 1ファイル |
| gradient checkpointing導入（train.py） | 4h | 1ファイル |
| gradient accumulation対応（train.py） | 3h | 1ファイル |
| 動作検証・VRAM計測 | 4h | - |
| **小計** | **12h（1.5人日）** | |

### Phase 2: bfloat16対応

| タスク | 工数 | 変更ファイル |
|-------|------|------------|
| train.pyのautocast/GradScaler改修 | 4h | 1ファイル |
| configs/v2/*.jsonにbf16_runフラグ追加 | 1h | 2ファイル |
| config.pyにGPU bfloat16サポート検出追加 | 2h | 1ファイル |
| fp16フォールバック動作の検証 | 3h | - |
| **小計** | **10h（1.25人日）** | |

### Phase 3: TextEncoder拡張

| タスク | 工数 | 変更ファイル |
|-------|------|------------|
| configs変更（n_heads, n_layers） | 1h | 2ファイル |
| attentions.pyの検証（4ヘッド動作確認） | 2h | 1ファイル |
| 事前学習モデルの部分ロード対応 | 6h | 1ファイル |
| 事前学習の実行 | 72-168h (計算時間) | - |
| 品質評価（CER/MCD測定） | 8h | - |
| **小計** | **17h（2人日）+ 計算時間3-7日** | |

### Phase 4: LoRA統合

| タスク | 工数 | 変更ファイル |
|-------|------|------------|
| LoRAConv1dクラスの実装 | 8h | modules.py |
| models.pyへのLoRA適用箇所の組み込み | 8h | models.py |
| Speaker Embedding→LoRA行列生成の実装 | 6h | models.py |
| train.pyのLoRA学習対応（凍結制御等） | 6h | train.py |
| 推論パイプラインのLoRA対応 | 4h | pipeline.py |
| テスト・デバッグ | 8h | - |
| **小計** | **40h（5人日）** | |

### Phase 5: Causal Convolution

| タスク | 工数 | 変更ファイル |
|-------|------|------------|
| modules.py WNのcausalパディング実装 | 4h | modules.py |
| models.pyのcausalフラグ伝搬 | 3h | models.py |
| リアルタイム推論ツールとの統合テスト | 6h | rvc_for_realtime.py |
| 双方向/causalの品質比較テスト | 8h | - |
| **小計** | **21h（2.6人日）** | |

### 全体

| Phase | 工数 | 累積 |
|-------|------|------|
| Phase 1: セグメント長拡張 | 1.5人日 | 1.5人日 |
| Phase 2: bfloat16 | 1.25人日 | 2.75人日 |
| Phase 3: TextEncoder拡張 | 2人日 + 計算3-7日 | 4.75人日 + 計算時間 |
| Phase 4: LoRA | 5人日 | 9.75人日 |
| Phase 5: Causal Conv | 2.6人日 | 12.35人日 |
| **合計** | **約12.5人日（2.5週間）** | + 事前学習計算時間 |

---

## 期待される品質向上

### 定量的指標の改善予測

| 指標 | 現行ベースライン | Phase 1後 | Phase 1+2+3後 | 全Phase後 |
|-----|--------------|---------|-------------|---------|
| CER（文字誤り率） | ~15-20% | ~12-15% | ~8-12% | ~6-10% |
| MCD（メルケプストラム歪み） | ~7.0 dB | ~6.5 dB | ~6.0 dB | ~5.5 dB |
| 話者類似度 | ~0.75 | ~0.78 | ~0.80 | ~0.83 |
| UTMOS | ~3.5 | ~3.7 | ~3.9 | ~4.1 |

注: 上記は歌唱音声変換における概算予測値。データセット、話者、楽曲により大きく変動する。

### 改善が期待される具体的な問題

1. **子音の脱落**: 「か」が「あ」に聞こえる問題 → セグメント長拡大で子音-母音遷移の学習改善
2. **母音の混同**: 「い」と「え」の区別が不明瞭 → TextEncoderの4ヘッド化で母音特徴の分離向上
3. **ビブラート崩れ**: 短いセグメントでビブラートが途切れる → 1.0s以上で1-2周期を包含
4. **ブレス音の不自然さ**: フレーズ境界の処理改善 → 長セグメントでフレーズ構造を学習
5. **ピッチ追従の遅延**: リアルタイム処理時 → Causal Convolutionで低レイテンシ化

---

## 依存関係

### 外部依存

| 依存 | バージョン要件 | 用途 |
|------|-------------|------|
| PyTorch | >= 2.0（現行2.10） | bfloat16 autocast、gradient checkpointing |
| CUDA | >= 11.8 | bfloat16ネイティブサポート |
| GPU Driver | >= 525.xx | Ampere bf16対応 |

### 内部依存関係図

```
Phase 1 (segment_size拡張)
    ↓
Phase 2 (bfloat16) ← Phase 1とは独立して実施可能だが、
    ↓                 segment_size拡大時の安定性向上に寄与
Phase 3 (TextEncoder拡張) ← 事前学習のやり直しが必要
    ↓                         Phase 1, 2を含む設定で事前学習すべき
Phase 4 (LoRA) ← Phase 3の新TextEncoderアーキテクチャが前提
    ↓              LoRA適用箇所はTextEncoder拡張後に決定
Phase 5 (Causal Conv) ← 独立して実施可能だが、
                          品質比較にはPhase 1-3の安定後が望ましい
```

### データ依存

- 事前学習データ: 既存のデータセット（50時間のオープンソースデータ）に加え、日本語音声データ（ReazonSpeech等）が必要
- 評価データ: CER計測用の参照テキスト付き日本語歌唱データ
- Whisper large-v3: CER計測用（推論のみ）

---

## 推奨実装順序

### ステップ1（Week 1）: セグメント長拡張 + bfloat16

**目標**: 最小の変更で最大の効果を得る

1. `configs/v2/48k.json` の `segment_size` を `34560` に変更
2. `configs/v2/32k.json` の `segment_size` を `25600` に変更
3. `train.py` にgradient checkpointing（torch.utils.checkpoint）を追加
4. `train.py` にbfloat16対応を追加（Ampere以降は自動でbf16、それ以外はfp16フォールバック）
5. 既存の事前学習モデルで動作検証（segment_size変更は事前学習モデルの互換性に影響しない）
6. VRAM使用量とstep/secを計測

**この段階での品質改善予測: +15-20%（CER基準）**

### ステップ2（Week 1-2）: segment_sizeの段階的拡大検証

7. segment_size=48000で検証（1.0s）
8. segment_size=72000で検証（1.5s）
9. 各設定でCER/MCD/話者類似度を計測し、VRAM-品質のトレードオフを定量化
10. 最適なsegment_sizeをデフォルトとして決定

### ステップ3（Week 2-3）: TextEncoder拡張

11. n_heads=4、n_layers=8の設定に変更
12. 事前学習モデルの部分ロード機能を実装（既存のEncoder層は流用、追加分はランダム初期化）
13. 事前学習の実行（日本語データで3-7日）
14. 品質評価・比較

**この段階での品質改善予測: +25-30%（CER基準、ベースラインから）**

### ステップ4（Week 3-4）: LoRA統合

15. LoRAConv1dクラスの実装とユニットテスト
16. TextEncoder・PosteriorEncoder・Flowへの統合
17. Speaker Embeddingからの行列生成パスの実装
18. ファインチューニングワークフローの整備（本体凍結 + LoRAのみ学習）
19. 少量データ（5-10分）でのファインチューニング品質検証

### ステップ5（Week 4-5）: Causal Convolution（オプション）

20. modules.pyにcausalパディングオプションを追加
21. リアルタイム推論ツールとの統合
22. オフライン/オンラインそれぞれでの品質評価
23. リアルタイムモードでの遅延測定

### 各ステップの判断基準

| ステップ | 次に進む条件 | 中止条件 |
|---------|------------|---------|
| Step 1 | VRAM 12GB以内で安定学習 | loss発散、VRAM不足 |
| Step 2 | CERが現行比10%以上改善 | 品質改善が5%未満 |
| Step 3 | 事前学習後のCERが更に改善 | 事前学習リソースが確保不可 |
| Step 4 | 少量データFTで品質維持 | LoRA学習が不安定 |
| Step 5 | リアルタイム遅延が許容範囲 | 品質劣化が大きい |

---

## まとめ

本提案の核心は「セグメント長の拡大」であり、これは設定変更のみで実施可能な最もコストパフォーマンスの高い改善である。nadare氏の実証結果に基づき、34,560サンプル（0.72秒）への拡大を第一段階として推奨する。

TextEncoderの拡張やLoRA統合は中長期的な品質向上に寄与するが、事前学習のやり直しを伴うため、セグメント長拡大の効果を検証した後に着手すべきである。bfloat16への移行はセグメント長拡大と同時に実施することで、学習安定性の向上とVRAM管理の改善を両立できる。

Causal Convolutionはリアルタイム推論の改善に有効だが、オフライン処理が主な用途である場合は優先度を下げてよい。
