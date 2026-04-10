# ボコーダ/デコーダ アーキテクチャ改善提案書

> 対象: RVC WebUI (Retrieval-based Voice Conversion)
> 作成日: 2026-04-10
> 目的: 日本語歌声変換品質の向上

---

## 1. 概要

### 1.1 現状のデコーダアーキテクチャ

本プロジェクトのデコーダは **GeneratorNSF** であり、HiFi-GAN V1 にNeural Source Filter (NSF) を組み合わせた構成をとる。F0条件付き歌声合成に対応するが、設計は2020年のHiFi-GAN論文時点のものがほぼそのまま使われている。

**現行構成の詳細:**

| 要素 | 現状の値 | ファイル位置 |
|------|---------|------------|
| Generator本体 | `GeneratorNSF` | `infer/lib/infer_pack/models.py` L423-555 |
| 活性化関数 | `LeakyReLU(slope=0.1)` | `infer/lib/infer_pack/modules.py` L15 (`LRELU_SLOPE = 0.1`) |
| アップサンプル率 | `[12, 10, 2, 2]` (積=480=hop_length, 48kHz時) | `configs/v2/48k.json` |
| アップサンプル初期チャンネル | `512` | `configs/v2/48k.json` |
| ResBlock | `ResBlock1` (dilation `[[1,3,5],[1,3,5],[1,3,5]]`) | `infer/lib/infer_pack/modules.py` L231-337 |
| ResBlockカーネル | `[3, 7, 11]` | `configs/v2/48k.json` |
| アップサンプルカーネル | `[24, 20, 4, 4]` | `configs/v2/48k.json` |
| mel_fmin | `0.0` (DC成分を含む) | `configs/v2/48k.json` |
| mel_fmax | `null` (ナイキスト周波数まで) | `configs/v2/48k.json` |
| NSFソース | `SourceModuleHnNSF` (SineGen, harmonic_num=0) | `infer/lib/infer_pack/models.py` L368-420 |
| Discriminator | `MultiPeriodDiscriminatorV2` (periods=[2,3,5,7,11,17,23,37] + DiscriminatorS) | `infer/lib/infer_pack/models.py` L892-917 |
| アンチエイリアス | なし | - |
| 損失関数 | L1 mel loss + feature matching + GAN loss + KL loss | `infer/lib/train/losses.py` |

### 1.2 現行アーキテクチャの問題点

1. **LeakyReLUは周期的信号に不適**: 音声波形は本質的に周期的だが、LeakyReLUには周期性に関する帰納バイアスがない。高音域の歌声（特に女声・ファルセット）で倍音構造の再現が不正確になる。
2. **アンチエイリアスフィルタの欠如**: `ConvTranspose1d` によるアップサンプリング時にエイリアシングノイズが発生し、高周波アーティファクトの原因となる。
3. **DC成分の包含**: `mel_fmin=0.0` により0Hz付近の無意味な情報がモデルに入力され、有効なmel帯域を浪費している。
4. **ディスクリミネータの限界**: MPD+MSDの組み合わせは時間方向の周期性を捉えるが、周波数方向のマルチスケール構造（倍音列など）の評価が弱い。
5. **固定的な低効率アーキテクチャ**: HiFi-GAN V1の逐次畳み込み方式は、iSTFTベースの手法と比較して計算効率が大幅に劣る。

### 1.3 改善の方針

既存の事前学習済みモデルとの互換性、段階的な導入可能性、コスト対効果を考慮し、9段階の改善案を侵襲度の低い順に提案する。Phase 1-3は現行モデルの再学習のみで対応可能、Phase 4以降はアーキテクチャ変更を伴う。

---

## 2. 技術的詳細

### 2.1 SnakeBeta活性化関数 (BigVGAN由来)

**変更対象:** `infer/lib/infer_pack/modules.py` の `ResBlock1`, `ResBlock2` 内の `F.leaky_relu` 呼び出し、および `infer/lib/infer_pack/models.py` の `GeneratorNSF.forward` 内の `F.leaky_relu` 呼び出し。

**現行コード (modules.py L307-316):**
```python
def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None):
    for c1, c2 in zip(self.convs1, self.convs2):
        xt = F.leaky_relu(x, self.lrelu_slope)
        ...
        xt = F.leaky_relu(xt, self.lrelu_slope)
```

**提案する変更:**
```python
class SnakeBeta(nn.Module):
    """BigVGAN の Snake activation: x + (1/a) * sin^2(a * x)
    Beta変種: x + (1/b) * sin^2(a * x) (a, b は独立に学習)
    """
    def __init__(self, channels, alpha_logscale=True):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))
        self.alpha_logscale = alpha_logscale
        # 初期値: alpha=1.0, beta=1.0 に相当
        nn.init.ones_(self.alpha)
        nn.init.ones_(self.beta)

    def forward(self, x):
        alpha = self.alpha.exp() if self.alpha_logscale else self.alpha
        beta = self.beta.exp() if self.alpha_logscale else self.beta
        # x + (1/beta) * sin^2(alpha * x)
        return x + (1.0 / beta) * torch.sin(alpha * x).pow(2)
```

**理論的根拠:**
- `sin^2(ax)` の項が周期的帰納バイアスを導入し、倍音構造のモデリングを促進する。
- パラメータ `a` がデータから学習されるため、異なるチャンネルが異なる周波数帯域に特化できる。
- Seed-VCプロジェクトで高音域歌声の品質が大幅に改善されたことが確認済み。

**実装上の注意:**
- `ResBlock1` の各畳み込みの前に配置（`F.leaky_relu` を置換）。
- チャンネル次元でパラメータ化するため、各ResBlock層のチャンネル数に合わせてインスタンス化が必要。
- `GeneratorNSF.forward` 内のアップサンプル前後の `F.leaky_relu` も置換対象。

### 2.2 アンチエイリアスフィルタ

**変更対象:** `infer/lib/infer_pack/models.py` の `GeneratorNSF.__init__` 内のアップサンプル層 (`ConvTranspose1d`)。

**提案する追加モジュール:**
```python
class AntiAliasActivation(nn.Module):
    """Kaiser窓sinc LPFによるアンチエイリアス処理"""
    def __init__(self, channels, upsample_rate, filt_size=12, beta=14.769656459379492):
        super().__init__()
        self.filt_size = filt_size
        # Kaiser窓sinc LPFの設計
        cutoff = 1.0 / upsample_rate
        kernel = self._design_kaiser_lpf(filt_size, cutoff, beta)
        self.register_buffer('filt', kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1))
        self.channels = channels

    def _design_kaiser_lpf(self, N, cutoff, beta):
        """Kaiser窓sinc低域通過フィルタ"""
        n = torch.arange(N) - (N - 1) / 2
        sinc = torch.sinc(2 * cutoff * n)
        window = torch.kaiser_window(N, periodic=False, beta=beta)
        kernel = sinc * window
        return kernel / kernel.sum()

    def forward(self, x):
        # グループ畳み込みでチャンネル独立にフィルタリング
        pad = self.filt_size // 2
        x = F.pad(x, (pad, pad), mode='reflect')
        return F.conv1d(x, self.filt, groups=self.channels)
```

**挿入位置:** 各 `ConvTranspose1d` の直後、ResBlock処理の直前。

**理論的根拠:**
- `ConvTranspose1d` のアップサンプリングは本質的に0挿入+畳み込みであり、スペクトルの折り返し（イメージ成分）が発生する。
- LPFでナイキスト周波数以上を抑制することで、アーティファクトを除去。
- BigVGAN、Vocosなど最新ボコーダでは標準的に採用されている。

### 2.3 mel_fmin の変更

**変更対象:** `configs/v2/48k.json`, `configs/v2/32k.json` の `data.mel_fmin` フィールド。

**現行:** `"mel_fmin": 0.0`
**提案:** `"mel_fmin": 40.0`

**理論的根拠:**
- 人間の聴覚の下限は約20Hzだが、歌声の基本周波数は通常80Hz以上（男性低音域でも約80Hz）。
- 0-40Hz帯域にはDCオフセットや低周波ノイズしか含まれず、有効な音声情報はない。
- OpenVPIプロジェクト（DiffSinger等）で `mel_fmin=40.0` が実績あり。
- mel帯域を40Hz以上に限定することで、同じ `n_mel_channels=128` でもより有効な周波数分解能が得られる。

**影響範囲:**
- `infer/lib/train/mel_processing.py` の `mel_spectrogram_torch` 関数に影響。
- 設定変更のみで既存コードの修正は不要だが、**事前学習済みモデルとの互換性が失われるため再学習が必須**。

### 2.4 MRD (Multi-Resolution Discriminator)

**変更対象:** `infer/lib/infer_pack/models.py` に新クラス追加、`infer/modules/train/train.py` の学習ループを修正。

**提案するアーキテクチャ:**
```python
class MultiResolutionDiscriminator(nn.Module):
    """UnivNet / BigVGAN 方式の MRD"""
    def __init__(self, resolutions=None):
        super().__init__()
        if resolutions is None:
            # (n_fft, hop_size, win_length) のリスト
            resolutions = [
                (1024, 120, 600),
                (2048, 240, 1200),
                (512, 50, 240),
            ]
        self.discriminators = nn.ModuleList([
            DiscriminatorR(resolution) for resolution in resolutions
        ])

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class DiscriminatorR(nn.Module):
    """STFT-based single-resolution discriminator"""
    def __init__(self, resolution, channels=32, in_channels=1):
        super().__init__()
        self.resolution = resolution
        n_fft, hop_size, win_length = resolution
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(in_channels, channels, (3, 9), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 3), padding=(1, 1))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(channels, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        fmap = []
        # STFT magnitude spectrogram を入力として使用
        n_fft, hop_size, win_length = self.resolution
        mag = self._stft(x, n_fft, hop_size, win_length)
        x = mag.unsqueeze(1)
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap
```

**学習ループへの統合 (`infer/modules/train/train.py`):**
```python
# L171 付近に追加
net_d_mrd = MultiResolutionDiscriminator()

# L450-453 の Discriminator 処理に並列追加
y_d_hat_r_mrd, y_d_hat_g_mrd, _, _ = net_d_mrd(wave, y_hat.detach())
loss_disc_mrd, _, _ = discriminator_loss(y_d_hat_r_mrd, y_d_hat_g_mrd)
loss_disc_total = loss_disc + loss_disc_mrd

# L462-468 の Generator 処理に並列追加
y_d_hat_r_mrd, y_d_hat_g_mrd, fmap_r_mrd, fmap_g_mrd = net_d_mrd(wave, y_hat)
loss_fm_mrd = feature_loss(fmap_r_mrd, fmap_g_mrd)
loss_gen_mrd, _ = generator_loss(y_d_hat_g_mrd)
loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_gen_mrd + loss_fm_mrd
```

**重要:** MRDは学習時のみ使用。推論時は不要のため、推論コードに変更なし。

### 2.5 CQT Discriminator

**変更対象:** `infer/lib/infer_pack/models.py` に追加。

**提案するアーキテクチャ:**
```python
class CQTDiscriminator(nn.Module):
    """CQT(Constant-Q Transform)ベースのディスクリミネータ。
    倍音構造を等比間隔で分析するため、音楽・歌声に最適。
    """
    def __init__(self, hop_length=256, n_octaves=8, bins_per_octave=24,
                 sample_rate=48000, channels=32):
        super().__init__()
        self.cqt_params = {
            'hop_length': hop_length,
            'n_bins': n_octaves * bins_per_octave,
            'bins_per_octave': bins_per_octave,
            'sr': sample_rate,
        }
        n_bins = n_octaves * bins_per_octave
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, channels, (3, 9), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 3), padding=(1, 1))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(channels, 1, (3, 3), padding=(1, 1)))
```

**MRDとの違い:**
- STFTは等間隔周波数ビンだが、CQTは等比間隔（オクターブベース）で周波数を分解。
- 基本周波数に対する倍音列（1x, 2x, 3x, ...）をより自然に捉えられる。
- 歌声のビブラートや声区遷移の評価に適する。

### 2.6 upsample_initial_channel の増加

**変更対象:** `configs/v2/48k.json`, `configs/v2/32k.json`

**現行:** `"upsample_initial_channel": 512`
**提案:** `"upsample_initial_channel": 768`

**影響:**
- GeneratorNSF内の全アップサンプル層とResBlockのチャンネル数が比例して増加。
- 各段のチャンネル数: 768 -> 384 -> 192 -> 96 -> 48（現行: 512 -> 256 -> 128 -> 64 -> 32）。
- パラメータ数は約2.25倍に増加。
- `conv_pre` の出力チャンネル、`noise_convs` のチャンネル、ResBlockの全畳み込み層に影響。
- `cond` 層 (`nn.Conv1d(gin_channels, upsample_initial_channel, 1)`) も自動的に拡大。

### 2.7 Full BigVGAN Migration

**変更対象:** `infer/lib/infer_pack/models.py` の `GeneratorNSF` クラス全体を置換。

**主要な変更点:**

| コンポーネント | 現行 (HiFi-GAN + NSF) | BigVGAN + NSF |
|---------------|----------------------|---------------|
| 活性化関数 | LeakyReLU | SnakeBeta |
| ResBlock | `ResBlock1` (Conv1d + dilation) | AMPBlock (Anti-aliased Multi-Periodicity Block) |
| アップサンプル | `ConvTranspose1d` 素通し | `ConvTranspose1d` + Kaiser LPF |
| 正規化 | weight_norm | weight_norm (同一) |
| conv_post活性化 | `tanh` | `tanh` (同一) |
| NSFソース | 保持 | 保持（BigVGAN固有部分にNSFを統合） |

**AMPBlock の構造:**
```python
class AMPBlock(nn.Module):
    """Anti-aliased Multi-Periodicity Block (BigVGAN)
    各dilated conv の前後にアンチエイリアスフィルタを挿入。
    """
    def __init__(self, channels, kernel_size, dilation, activation=SnakeBeta):
        super().__init__()
        self.activation_pre = activation(channels)
        self.conv1 = weight_norm(Conv1d(channels, channels, kernel_size,
                                        dilation=dilation,
                                        padding=get_padding(kernel_size, dilation)))
        self.aa_filter1 = AntiAliasActivation(channels, upsample_rate=1)
        self.activation_mid = activation(channels)
        self.conv2 = weight_norm(Conv1d(channels, channels, kernel_size,
                                        dilation=1,
                                        padding=get_padding(kernel_size, 1)))
        self.aa_filter2 = AntiAliasActivation(channels, upsample_rate=1)
```

### 2.8 Vocos (ConvNeXt + iSTFT)

**変更対象:** `GeneratorNSF` を完全に新アーキテクチャで置換。

**アーキテクチャ概要:**
```
入力特徴量 (192ch) -> ConvNeXt Backbone -> iSTFT Head -> 波形出力

ConvNeXt Backbone:
  - ConvNeXtBlock x N (depthwise separable conv + GeLU + LayerNorm)
  - 全ての処理がmelスケール（低時間解像度）で実行
  - アップサンプルは最終段のiSTFTで一括実行

iSTFT Head:
  - magnitude予測 (Linear -> exp)
  - phase予測 (Linear -> atan2)
  - torch.istft() で波形復元
```

**現行との構造的差異:**

| 項目 | 現行 GeneratorNSF | Vocos |
|------|-------------------|-------|
| アップサンプル方式 | 逐次ConvTranspose (4段) | iSTFT一括 |
| 中間表現 | 時間領域波形（段階的解像度向上） | 周波数領域（mel解像度で全処理） |
| NSF統合 | har_source加算（各段） | 位相予測にF0情報を注入 |
| 推論速度 | 1x (基準) | 約13x高速 |
| パラメータ数 | 約14M (512ch時) | 約13M |

**NSFとの統合方法:**
```python
class VocosNSFHead(nn.Module):
    """iSTFT + NSF ハイブリッドヘッド
    magnitudeは学習予測、phaseはNSFの正弦波から初期推定して残差学習。
    """
    def __init__(self, n_fft, hop_length, sr):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.source = SourceModuleHnNSF(sr, harmonic_num=0)
        self.mag_head = nn.Linear(hidden_dim, n_fft // 2 + 1)
        self.phase_residual = nn.Linear(hidden_dim, n_fft // 2 + 1)

    def forward(self, x, f0):
        # NSFソースからの位相初期値
        har_source, _, _ = self.source(f0, self.hop_length)
        source_phase = torch.angle(torch.stft(har_source, self.n_fft, ...))

        mag = self.mag_head(x).exp()
        phase_res = self.phase_residual(x)
        phase = source_phase + phase_res  # 残差学習

        # iSTFT
        stft = mag * torch.exp(1j * phase)
        wav = torch.istft(stft, self.n_fft, hop_length=self.hop_length)
        return wav
```

### 2.9 HiFTNet (iSTFT + Harmonic-plus-Noise)

**変更対象:** `GeneratorNSF` を置換。

**アーキテクチャ概要:**
```
入力特徴量 -> HiFi-GAN-like Backbone (低アップサンプル率)
                     |
                     v
              Source Generator (Harmonic + Noise)
                     |
                     v
              iSTFT-based Synthesis
                     |
                     v
                  波形出力
```

**HiFTNetの設計思想:**
- HiFi-GANの前段部分（特徴量→中間表現）は保持しつつ、最終段のアップサンプルをiSTFTに置換。
- Harmonic-plus-Noiseソースフィルタを明示的に組み込み（現行のNSFと類似だがより洗練）。
- BigVGANの1/6のパラメータ数で同等以上の品質。

**現行NSFとの統合の容易さ:**
- 既存の `SourceModuleHnNSF` と設計思想が共通しているため、移行は比較的容易。
- SineGen部分はそのまま再利用可能。

---

## 3. コスト/効果比較表

### 3.1 総合比較

| # | 改善案 | 品質向上 | 推論速度 | パラメータ増 | コード変更量 | 再学習コスト | 互換性破壊 | 総合評価 |
|---|--------|---------|---------|-------------|-------------|-------------|-----------|---------|
| 1 | SnakeBeta活性化 | ★★★★ | ±0 | +0.1% | 小 (約100行追加) | 全再学習 | あり | **最優先** |
| 2 | アンチエイリアスフィルタ | ★★★ | -5%程度 | +0.5% | 小 (約80行追加) | 全再学習 | あり | 高 |
| 3 | mel_fmin変更 | ★★ | ±0 | ±0 | 極小 (設定のみ) | 全再学習 | あり | **最低コスト** |
| 4 | MRD追加 | ★★★ | ±0 (学習のみ) | +3M (学習時) | 中 (約200行) | 全再学習 | なし | 高 |
| 5 | CQT Discriminator | ★★★ | ±0 (学習のみ) | +2M (学習時) | 中 (約200行) | 全再学習 | なし | 中 |
| 6 | upsample_initial_channel増 | ★★ | -30% | +125% | 極小 (設定のみ) | 全再学習 | あり | 低 |
| 7 | Full BigVGAN | ★★★★★ | -10% | +20% | 大 (約500行) | 全再学習 | あり | 高(長期) |
| 8 | Vocos | ★★★★ | +1200% | -10% | 大 (約600行新規) | 全再学習 | あり | 中(実験的) |
| 9 | HiFTNet | ★★★★★ | +300% | -80% | 大 (約500行新規) | 全再学習 | あり | **最終目標** |

### 3.2 VRAM要件

| # | 改善案 | 学習時VRAM (batch=4) | 推論時VRAM | 備考 |
|---|--------|---------------------|-----------|------|
| 0 | 現行 (参考) | 約6-8 GB | 約2-3 GB | RTX 3060以上 |
| 1 | SnakeBeta活性化 | 約6-8 GB (+0.1GB) | 約2-3 GB | 現行とほぼ同じ |
| 2 | アンチエイリアスフィルタ | 約7-9 GB (+0.5GB) | 約2.5-3.5 GB | フィルタバッファ分 |
| 3 | mel_fmin変更 | 約6-8 GB (±0) | 約2-3 GB | 変化なし |
| 4 | MRD追加 | 約9-12 GB (+3GB) | 約2-3 GB | 学習時のみ増加 |
| 5 | CQT Discriminator | 約9-11 GB (+2GB) | 約2-3 GB | CQT計算のVRAM |
| 6 | upsample_initial 768 | 約12-16 GB (+6GB) | 約4-6 GB | **RTX 3090/4080以上推奨** |
| 7 | Full BigVGAN | 約10-14 GB (+4GB) | 約3-4 GB | RTX 3080以上推奨 |
| 8 | Vocos | 約6-8 GB (-1GB) | 約1-2 GB | iSTFTは軽量 |
| 9 | HiFTNet | 約5-7 GB (-2GB) | 約1-1.5 GB | 最も軽量 |

### 3.3 実装工数

| # | 改善案 | 実装工数 | 検証工数 | リスク |
|---|--------|---------|---------|--------|
| 1 | SnakeBeta活性化 | 1-2日 | 3-5日 (再学習) | 低: ドロップイン置換 |
| 2 | アンチエイリアスフィルタ | 2-3日 | 3-5日 (再学習) | 低: 追加モジュール |
| 3 | mel_fmin変更 | 0.5日 | 3-5日 (再学習) | 極低: 設定変更のみ |
| 4 | MRD追加 | 3-5日 | 5-7日 (再学習) | 低: 学習コードのみ |
| 5 | CQT Discriminator | 5-7日 | 5-7日 (再学習) | 中: CQTライブラリ依存 |
| 6 | upsample_initial 768 | 0.5日 | 5-10日 (再学習) | 中: VRAM制約 |
| 7 | Full BigVGAN | 7-10日 | 7-14日 (再学習) | 中: 大規模リファクタ |
| 8 | Vocos | 10-15日 | 14-21日 (再学習) | 高: 全く異なるパラダイム |
| 9 | HiFTNet | 7-10日 | 10-14日 (再学習) | 中: NSF設計思想が共通 |

---

## 4. メリット

### 4.1 音質改善

1. **高音域の忠実度向上**: SnakeBeta活性化とアンチエイリアスフィルタにより、女声や裏声の高音域（C5以上、約523Hz以上）で発生していた金属的なアーティファクト、ジリジリしたノイズが大幅に軽減される。日本語歌声特有のファルセットやヘッドボイスの品質が特に改善される。

2. **倍音構造の正確な再現**: CQTディスクリミネータとMRDの組み合わせにより、基本周波数に対する倍音列（第2倍音、第3倍音、...）の振幅比が原音に忠実になる。日本語の母音「い」「え」のフォルマント構造やビブラートの自然さが向上する。

3. **声区遷移の滑らかさ**: 胸声（地声）からヘッドボイス、ファルセットへの遷移時に発生していた不自然な音質の急変が、NSFソースフィルタとBigVGANの周期的活性化の組み合わせで緩和される。

4. **子音・過渡音の明瞭度**: MRDの短窓分解能（n_fft=512, hop=50）により、「さ行」「た行」「は行」などの無声子音や破裂子音の時間微細構造がより正確に再現される。

5. **DC/低周波ノイズの除去**: mel_fmin=40.0 への変更により、マイク由来のポップノイズやハム音成分がmelスペクトログラムに混入しなくなり、低域の音質が安定する。

### 4.2 性能改善

6. **推論速度の大幅向上 (Vocos/HiFTNet選択時)**: iSTFTベースのデコーダはConvTranspose1dの逐次アップサンプリングと比較して4-13倍高速。リアルタイム変換（`tools/rvc_for_realtime.py`）でのレイテンシが大幅に改善される。

7. **モデルサイズの削減 (HiFTNet選択時)**: パラメータ数が約1/6になり、モデルファイルのダウンロードサイズとロード時間が短縮される。

### 4.3 学習改善

8. **学習の安定化**: MRD/CQTディスクリミネータの追加により、Generatorが受けるフィードバックの多様性が増し、モード崩壊（特定の音質パターンへの収束）のリスクが低減する。

9. **少量データでの効率的な学習**: SnakeBetaの周期的帰納バイアスにより、モデルが倍音構造を「ゼロから学習する」必要がなくなり、10分程度の少量データでもより良好な結果が得られるようになる。

### 4.4 保守性

10. **段階的導入による低リスク**: 提案1-3は独立に導入・検証可能であり、効果が認められない場合はロールバックが容易。学習コードのみの変更（提案4-5）は推論側に一切影響しない。

---

## 5. デメリット・リスク

### 5.1 互換性

1. **既存モデルの非互換**: 提案1, 2, 6-9はモデルアーキテクチャの変更を伴い、既存の事前学習済みモデル（`assets/pretrained_v2/`）および全てのユーザー学習済みモデル（`assets/weights/`）が使用不可能になる。コミュニティに蓄積された大量のモデル資産が失われるリスクがある。

2. **v2との命名衝突**: 現行が「v2」であるため、新アーキテクチャには「v3」等の新バージョン体系が必要。コード内の `SynthesizerTrnMs768NSFsid` クラス名やconfig構造の見直しが必要。

### 5.2 技術的リスク

3. **NSFとの統合の不確実性**: BigVGAN, Vocos, HiFTNetはいずれもオリジナル論文ではNSFとの組み合わせを想定していない。RVCのF0条件付き生成にこれらを適用する際、NSFソースフィルタの注入方法に試行錯誤が必要。特にVocosのiSTFTヘッドにNSFの正弦波ソースをどう統合するかは未解決の研究課題。

4. **学習不安定性の増大**: MRD + CQT + MPD の3つのディスクリミネータを同時使用する場合、損失のバランス調整が困難になる。各ディスクリミネータの損失重みのハイパーパラメータチューニングが必要で、不適切な重み設定はGeneratorの学習崩壊を引き起こしうる。

5. **CQTの計算コスト**: CQT計算は再帰的なフィルタバンク処理を要し、GPU上での効率的な実装が非自明。`nnAudio` や `torchaudio` のCQT実装はバッチ処理時にメモリ使用量が大きく、VRAM不足のリスクがある。ライブラリ依存も増加する。

### 5.3 運用リスク

6. **再学習コスト**: 全ての提案は事前学習済みモデルの再学習を必要とする。48kHzモデルの事前学習にはA100 80GB x 4で約1-2週間、V100 32GB x 4で約3-4週間かかる。電力コストと計算資源の確保が必要。

7. **評価の困難さ**: 歌声品質の評価は客観指標（PESQ, SI-SNR等）と主観評価（MOS）の乖離が大きい。特に日本語歌声の品質評価は標準化されたベンチマークが存在せず、改善効果の定量的な比較が困難。

8. **SnakeBeta の ONNX/TorchScript 互換性**: `sin^2(ax)` 関数と学習可能パラメータの組み合わせは一部の推論エンジン（ONNX Runtime、TorchScript）で最適化が効かない場合がある。`models_onnx.py` にも対応する変更が必要。

9. **アンチエイリアスフィルタのレイテンシ**: リアルタイム変換時、Kaiser窓フィルタの畳み込みが各アップサンプル段で追加されるため、チャンク処理でのレイテンシが増加する。`rvc_for_realtime.py` のバッファサイズ調整が必要。

10. **コミュニティへの影響**: RVCコミュニティは「アルゴリズム変更は基本的に受け付けていない」というポリシーを持つ（CLAUDE.md記載）。大規模なアーキテクチャ変更は上流リポジトリへの貢献が難しく、フォークとしてのメンテナンスコストが永続的に発生する。

---

## 6. 依存関係

### 6.1 提案間の依存関係マップ

```
[3] mel_fmin変更 ──(独立)
        |
[1] SnakeBeta ──────────────┐
        |                    |
[2] アンチエイリアス ────────┼──> [7] Full BigVGAN
        |                    |
[4] MRD ──(独立)             |
        |                    |
[5] CQT Disc ──(独立)       |
                             |
[6] upsample_initial ──(独立,任意の組み合わせ可)

[8] Vocos ──(独立パス、1-7と排他)
[9] HiFTNet ──(独立パス、1-7と排他)
```

**依存関係の詳細:**

| 提案 | 前提条件 | 併用推奨 | 排他関係 |
|------|---------|---------|---------|
| 1 SnakeBeta | なし | 2 (相乗効果) | なし |
| 2 アンチエイリアス | なし | 1 (相乗効果) | なし |
| 3 mel_fmin | なし | なし | なし |
| 4 MRD | なし | 5 (併用推奨) | なし |
| 5 CQT Disc | 4 (先に導入推奨) | 4 | なし |
| 6 upsample_initial | なし | 1+2 (容量活用) | なし |
| 7 Full BigVGAN | 1+2 (包含) | 4+5 | 8, 9 |
| 8 Vocos | なし (完全独立) | 4+5 | 7, 9 |
| 9 HiFTNet | なし (完全独立) | 4+5 | 7, 8 |

### 6.2 外部ライブラリ依存

| 提案 | 追加ライブラリ | バージョン要件 | PyPI利用可否 |
|------|---------------|--------------|-------------|
| 1-4, 6 | なし（PyTorch標準機能のみ） | - | - |
| 5 | `nnAudio` または `torchaudio` (CQT) | nnAudio>=0.3.2 | あり |
| 7 | なし（自前実装） | - | - |
| 8 | なし（自前実装、iSTFTはPyTorch標準） | torch>=2.0 | - |
| 9 | なし（自前実装） | torch>=2.0 | - |

---

## 7. 期待される品質向上

### 7.1 課題別の改善マッピング

| 現行の品質課題 | 主要な改善提案 | 期待改善度 |
|---------------|--------------|-----------|
| 高音域のジリジリしたノイズ | 1 (SnakeBeta) + 2 (AA) | 大幅改善 |
| ファルセットの不自然さ | 1 (SnakeBeta) + 7 (BigVGAN) | 大幅改善 |
| 「さ行」子音の劣化 | 4 (MRD) + 2 (AA) | 中程度改善 |
| ビブラートの不安定さ | 5 (CQT) + 1 (SnakeBeta) | 中程度改善 |
| 声区遷移のブツ切れ | 7 (BigVGAN) or 9 (HiFTNet) | 大幅改善 |
| 低域のモヤモヤ感 | 3 (mel_fmin) | 軽度改善 |
| 全体の解像感不足 | 6 (channel増) + 4 (MRD) | 中程度改善 |
| 推論レイテンシ | 8 (Vocos) or 9 (HiFTNet) | 4-13倍高速化 |

### 7.2 定量的な改善予測

| 指標 | 現行推定 | Phase 1 後 | Phase 2 後 | Phase 3 後 |
|------|---------|-----------|-----------|-----------|
| mel-cepstral distortion (dB) | ~5.5 | ~4.8 (-13%) | ~4.2 (-24%) | ~3.8 (-31%) |
| F0 RMSE (cent) | (デコーダ由来の劣化) | -5%程度 | -10%程度 | -15%程度 |
| 高周波帯域 (8-24kHz) SNR | ~15dB | ~19dB | ~22dB | ~25dB |
| リアルタイム係数 (RTF) | ~0.3 | ~0.3 | ~0.3 | ~0.07 (HiFTNet) |

※ 上記は類似プロジェクト（BigVGAN論文、Vocos論文、Seed-VC実験結果）からの外挿推定値であり、実測値ではない。

---

## 8. 推奨実装順序

### Phase 1: 低リスク高効果（推定 1-2週間）

```
Step 1a: mel_fmin = 0.0 -> 40.0 に変更 [提案3]
  - 変更: configs/v2/48k.json, configs/v2/32k.json
  - 工数: 0.5日
  - 検証: mel spectrogram の視覚的確認 + 短時間学習

Step 1b: SnakeBeta 活性化関数の導入 [提案1]
  - 変更: infer/lib/infer_pack/modules.py (SnakeBetaクラス追加、ResBlock1/2のforward修正)
  - 変更: infer/lib/infer_pack/models.py (GeneratorNSF/Generatorのforward内のleaky_relu置換)
  - 工数: 1-2日
  - 検証: 短時間学習 + 高音域の聴取評価

Step 1c: アンチエイリアスフィルタの追加 [提案2]
  - 変更: infer/lib/infer_pack/models.py (GeneratorNSF.__init__にフィルタ追加)
  - 工数: 2-3日
  - 検証: スペクトログラム上のエイリアシングアーティファクト確認

  -> Phase 1 完了後: 事前学習済みモデルの再学習実行
  -> 評価: MCD, 高周波SNR, 聴取評価
```

### Phase 2: ディスクリミネータ強化（推定 2-3週間）

```
Step 2a: MRD (Multi-Resolution Discriminator) の追加 [提案4]
  - 変更: infer/lib/infer_pack/models.py (MRDクラス追加)
  - 変更: infer/modules/train/train.py (学習ループにMRD統合)
  - 工数: 3-5日
  - 検証: Phase 1 のモデルと比較してA/Bテスト

Step 2b: CQT Discriminator の追加 [提案5]
  - 変更: infer/lib/infer_pack/models.py (CQTDiscクラス追加)
  - 変更: pyproject.toml (nnAudio依存追加)
  - 工数: 5-7日
  - 検証: 歌声のビブラート・倍音構造の改善度確認

  -> Phase 2 完了後: 事前学習済みモデルの再学習実行
  -> 評価: Phase 1 結果との比較、子音明瞭度の主観評価
```

### Phase 3: アーキテクチャ刷新（推定 4-6週間）

**2つの選択肢から1つを選ぶ:**

```
選択肢A: Full BigVGAN + NSF [提案7]
  - 推奨条件: 品質最優先、推論速度は現行と同程度で可
  - 変更: GeneratorNSF全体をAMPBlock + SnakeBeta + AAフィルタで再構成
  - Phase 1 の成果を包含するため、Step 1b, 1c の個別変更は不要になる
  - 工数: 7-10日 + 再学習 7-14日

選択肢B: HiFTNet + NSF [提案9]
  - 推奨条件: 品質と推論速度の両立、リアルタイム変換の大幅高速化
  - 変更: GeneratorNSFの後段をiSTFTベースに置換
  - 既存のSourceModuleHnNSFを再利用可能
  - 工数: 7-10日 + 再学習 10-14日
```

### Phase 4 (オプション): 実験的評価

```
Step 4: Vocos の実験的評価 [提案8]
  - Phase 3 で選択しなかった場合の代替評価
  - ブランチを分けて並行開発
  - ConvNeXt + iSTFT + NSF の統合実験
```

### 推奨ハードウェア構成

| Phase | 最低構成 | 推奨構成 | 事前学習時間 |
|-------|---------|---------|-------------|
| 1 | RTX 3060 12GB x1 | RTX 4080 16GB x1 | 3-5日 |
| 2 | RTX 3080 10GB x2 | RTX 4090 24GB x2 | 5-7日 |
| 3 | RTX 4090 24GB x2 | A100 80GB x4 | 7-14日 |

---

## 9. 実装時の具体的変更箇所まとめ

### 影響を受けるファイル一覧

| ファイル | Phase 1 | Phase 2 | Phase 3 |
|---------|---------|---------|---------|
| `configs/v2/48k.json` | mel_fmin変更 | - | upsample_initial等 |
| `configs/v2/32k.json` | mel_fmin変更 | - | upsample_initial等 |
| `infer/lib/infer_pack/modules.py` | SnakeBeta追加, ResBlock修正 | - | AMPBlock追加 (BigVGAN時) |
| `infer/lib/infer_pack/models.py` | GeneratorNSF修正, AAフィルタ追加 | MRD/CQT追加 | Generator全面改修 |
| `infer/lib/infer_pack/models_onnx.py` | 同期修正 | - | 同期修正 |
| `infer/modules/train/train.py` | - | MRD/CQT統合 | Generator対応 |
| `infer/lib/train/losses.py` | - | (変更不要) | - |
| `infer/lib/train/mel_processing.py` | (変更不要) | - | (変更不要) |
| `infer/modules/vc/modules.py` | - | - | 新Generator対応 |
| `infer/modules/vc/pipeline.py` | - | - | 新Generator対応 |
| `pyproject.toml` | - | nnAudio追加 (CQT時) | - |

### 設定ファイルの変更例 (Phase 1完了時)

```json
{
  "data": {
    "mel_fmin": 40.0
  },
  "model": {
    "activation": "snakebeta",
    "use_antialias": true
  }
}
```

### 設定ファイルの変更例 (Phase 3完了時、HiFTNet選択)

```json
{
  "model": {
    "generator_type": "hiftnet_nsf",
    "activation": "snakebeta",
    "use_antialias": true,
    "istft_n_fft": 16,
    "istft_hop_length": 4,
    "source_harmonic_num": 8
  }
}
```

---

## 10. 結論

現行のHiFi-GAN V1 + NSFデコーダは2020年時点では先進的な設計であったが、2024-2025年の研究進展により、特に歌声合成の文脈では複数の明確な改善余地が存在する。

**最も費用対効果の高い最初のステップ**は、Phase 1 の3つの変更（mel_fmin修正、SnakeBeta活性化、アンチエイリアスフィルタ）を同時に適用し、事前学習済みモデルを再学習することである。これだけで高音域の品質が体感的に大幅に改善されることが、Seed-VCやBigVGANの先行事例から見込まれる。

**長期的な最終目標**としては、HiFTNet + NSF の統合が最も有望である。現行NSFと設計思想が共通しており移行が比較的容易でありながら、推論速度4倍・パラメータ数1/6という劇的な効率化を実現できる。

ただし、全ての変更は既存モデルとの互換性を破壊するため、「v3」としての新バージョン体系の確立とコミュニティへの丁寧な移行パスの提供が不可欠である。
