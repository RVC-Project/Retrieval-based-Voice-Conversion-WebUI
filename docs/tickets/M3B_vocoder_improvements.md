# M3-B: ボコーダ改善

## メタ情報
- **マイルストーン**: M3
- **フェーズ**: Phase 3-B（Week 7-8）
- **工数見積もり**: 8-9人日 + GPU 14-30h（事前学習再実行）。内訳: SnakeBeta 3-4日 + アンチエイリアス 2日 + MRD/CQT 2日 + 事前学習再実行 1日。※milestones.md Phase 3-Bヘッダーには「7人日」と記載されているが、サブタスク合計と不整合（修正推奨）
- **GPU要件**: RTX 4090 24GB / Cloud GPU（事前学習再実行）
- **前提タスク**: M3-A完了（損失関数改善）、M2の事前学習済みモデル（kushinada + ContentVec）
- **ステータス**: 未着手
- **関連マイルストーン**: [milestones.md](../milestones.md) > M3 Phase 3-Bセクション
- **関連提案書**: [proposal_vocoder_decoder_improvements.md](../proposal_vocoder_decoder_improvements.md)

---

## 1. タスク目的とゴール

### 目的

現在のRVC WebUIのデコーダ（`GeneratorNSF`）は、2020年のHiFi-GAN V1にNeural Source Filter（NSF）を組み合わせた構成である。活性化関数には周期性に関する帰納バイアスを持たない `LeakyReLU(slope=0.1)` を使用し、アップサンプリング時のアンチエイリアス処理も欠如している。このため、高音域の歌声（特に女声・ファルセット、C5以上 = 約523Hz以上）で倍音構造の再現が不正確になり、金属的なアーティファクトやジリジリしたノイズが発生する。

本タスクでは、BigVGAN由来のSnakeBeta活性化関数とアンチエイリアスフィルタを導入してGenerator側の波形生成品質を向上させるとともに、MRD/CQTディスクリミネータを追加して学習時のフィードバック品質を改善する。加えて、M1から延期されていた `mel_fmin` の変更（0Hz→40Hz）もここで適用する。

### 達成基準

1. **SnakeBeta活性化関数**: `ResBlock1` / `ResBlock2` および `GeneratorNSF.forward` 内の全 `F.leaky_relu` をSnakeBeta活性化に置換し、ファインチューニングで高音域の音質改善を確認
2. **アンチエイリアスフィルタ**: 各 `ConvTranspose1d` の直後にKaiser窓sinc LPFを挿入し、高周波アーティファクトを抑制
3. **mel_fmin変更**: `configs/v2/48k.json` と `configs/v2/32k.json` の `mel_fmin` を `0.0` から `40.0` に変更し、DC/低周波ノイズを除去
4. **MRD/CQTディスクリミネータ**: 学習ループに追加し、周波数方向のマルチスケール構造評価を強化
5. **事前学習再実行**: 上記変更を反映した新しい事前学習済みモデルを生成（kushinada + SnakeBeta + アンチエイリアス + mel_fmin=40）

### Go/No-Go判定③ (Week 8終了時)

| 基準 | Go（M4へ進む） | M3で完了（M4スキップ） |
|------|----------------|----------------------|
| MCD  | < 6.5 dB       | M3成果で十分          |
| CER  | < 12%          | 主観品質が十分         |
| MOS  | > 3.8          | > 3.5で実用レベル      |

### 非ゴール

- GeneratorNSFのアーキテクチャ全面置換（Vocos/HiFTNet等はM4以降で検討）
- `upsample_initial_channel` の変更（512→768はVRAM増加が大きくM3では実施しない）
- コミュニティモデルとの後方互換性維持（本タスクは事前学習再実行を前提とする）

---

## 2. 実装する内容の詳細

### 2-1. タスク3-6: SnakeBeta活性化関数導入（3-4日）

**目的**: HiFi-GANの `LeakyReLU` を、BigVGAN由来の周期的活性化関数 `SnakeBeta` に置換する。同時に `mel_fmin=40.0` を適用する。

#### 2-1a. SnakeBeta クラスの実装

**新規追加先**: `infer/lib/infer_pack/modules.py`

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
        nn.init.ones_(self.alpha)
        nn.init.ones_(self.beta)

    def forward(self, x):
        alpha = self.alpha.exp() if self.alpha_logscale else self.alpha
        beta = self.beta.exp() if self.alpha_logscale else self.beta
        return x + (1.0 / beta) * torch.sin(alpha * x).pow(2)
```

**理論的根拠**:
- `sin^2(ax)` の項が周期的帰納バイアスを導入し、倍音構造のモデリングを促進する
- パラメータ `alpha` がデータから学習されるため、異なるチャンネルが異なる周波数帯域に特化できる
- パラメータ増加は `2 * channels` 個のスカラーのみ（全体の+0.1%未満）

#### 2-1b. ResBlock1 の改修

**変更対象**: `infer/lib/infer_pack/modules.py` L231-337 の `ResBlock1` クラス

**現行コード** (L307-320):
```python
def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None):
    for c1, c2 in zip(self.convs1, self.convs2):
        xt = F.leaky_relu(x, self.lrelu_slope)      # ← 置換対象
        if x_mask is not None:
            xt = xt * x_mask
        xt = c1(xt)
        xt = F.leaky_relu(xt, self.lrelu_slope)      # ← 置換対象
        if x_mask is not None:
            xt = xt * x_mask
        xt = c2(xt)
        x = xt + x
```

**変更方針**:
- `__init__` に `activation` パラメータを追加（デフォルト: `SnakeBeta`）
- `F.leaky_relu` 呼び出しをインスタンス化された `SnakeBeta` モジュールに置換
- SnakeBetaはチャンネル次元でパラメータ化するため、各ResBlock層のチャンネル数 `channels` を使って `SnakeBeta(channels)` をインスタンス化
- `convs1` と `convs2` それぞれの前に個別の `SnakeBeta` インスタンスが必要（合計 `2 * len(dilation)` 個 = 6個/ResBlock1）

```python
class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        # ... convs1, convs2 の定義は同一 ...

        # SnakeBeta活性化（各畳み込みの前に配置）
        self.activations1 = nn.ModuleList([
            SnakeBeta(channels) for _ in range(len(dilation))
        ])
        self.activations2 = nn.ModuleList([
            SnakeBeta(channels) for _ in range(len(dilation))
        ])

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None):
        for c1, c2, act1, act2 in zip(
            self.convs1, self.convs2, self.activations1, self.activations2
        ):
            xt = act1(x)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = act2(xt)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x
```

**`ResBlock2` にも同様の変更を適用**（`infer/lib/infer_pack/modules.py` L340-390）。ResBlock2は `convs` のみ（convs1/convs2の分離なし）で、dilation数が2のため `SnakeBeta` は2個。

#### 2-1c. GeneratorNSF.forward の改修

**変更対象**: `infer/lib/infer_pack/models.py` L423-555 の `GeneratorNSF` クラス

**現行コード** (L512-532):
```python
for i, (ups, noise_convs) in enumerate(zip(self.ups, self.noise_convs)):
    if i < self.num_upsamples:
        x = F.leaky_relu(x, self.lrelu_slope)   # ← 置換対象 (アップサンプル前, slope=0.1)
        x = ups(x)
        x_source = noise_convs(har_source)
        x = x + x_source
        # ... ResBlock処理 (L518-529) ...
x = F.leaky_relu(x)                              # ← 置換対象 (conv_post前, slope=デフォルト0.01)
x = self.conv_post(x)
x = torch.tanh(x)
```

> **注意**: ループ内の `F.leaky_relu(x, self.lrelu_slope)` は slope=0.1 だが、`conv_post` 前の `F.leaky_relu(x)` は引数なしでデフォルトの slope=0.01 を使用している。SnakeBeta置換後はこの差異は消滅する。

**変更方針**:
- `__init__` に各アップサンプル段用の `SnakeBeta` インスタンスを追加（`num_upsamples` 個）
- `conv_post` 前の `F.leaky_relu` 用に追加で1個
- `lrelu_slope` 属性は残す（後方互換のため。ただし `forward` では未使用になる）

```python
# __init__ に追加
self.act_pre = nn.ModuleList([
    SnakeBeta(upsample_initial_channel // (2**i))
    for i in range(self.num_upsamples)
])
ch_final = upsample_initial_channel // (2**self.num_upsamples)
self.act_post = SnakeBeta(ch_final)

# forward の変更
for i, (ups, noise_convs) in enumerate(zip(self.ups, self.noise_convs)):
    if i < self.num_upsamples:
        x = self.act_pre[i](x)              # SnakeBeta
        x = ups(x)
        # ...
x = self.act_post(x)                        # SnakeBeta
x = self.conv_post(x)
```

#### 2-1d. Generator (nono版) の改修

**変更対象**: `infer/lib/infer_pack/models.py` L193-289 の `Generator` クラス（F0なしモデル用、`SynthesizerTrnMs768NSFsid_nono` の `self.dec` で使用）

`GeneratorNSF` と同様に `F.leaky_relu` を `SnakeBeta` に置換する。`Generator.forward` (L237-266) 内のアップサンプルループおよび `conv_post` 前の活性化が対象。

> **注意**: `Generator.forward` (L253) では `F.leaky_relu(x, modules.LRELU_SLOPE)` でモジュール定数を直接参照しているのに対し、`GeneratorNSF.forward` (L514) では `self.lrelu_slope` インスタンス属性を使用している。両クラスで置換方法は同一だが、参照箇所が異なる点に注意。

#### 2-1e. mel_fmin 変更

**変更対象**: `configs/v2/48k.json` L25、`configs/v2/32k.json` L25

**現行**:
```json
"mel_fmin": 0.0
```

**変更後**:
```json
"mel_fmin": 40.0
```

**影響範囲**:
- `infer/lib/train/mel_processing.py` L96: `librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)` の `fmin` 引数に影響。なお、`spec_to_mel_torch` 内のmelフィルタバンクキャッシュ (L94) はキーに `fmax` のみを含み `fmin` を含まないため、同一プロセス内で `fmin` を動的に変更すると古いキャッシュが使われるバグがある。本タスクではconfigレベルの変更なのでプロセス再起動により問題は回避されるが、将来的にはキャッシュキーに `fmin` を含めるべき
- `infer/modules/train/train.py` L431, L443: `hps.data.mel_fmin` として2箇所で参照（`spec_to_mel_torch` と `mel_spectrogram_torch` の呼び出し）
- `infer/lib/rmvpe.py` L394: RMVPEでも `mel_fmin=0` がハードコードされているが、これはF0抽出用でありボコーダのmel_fminとは独立（変更不要）
- 設定変更のみでPythonコードの修正は不要
- **事前学習済みモデルとの互換性が完全に失われるため、再学習が必須**

> **注意**: mel_fmin変更はM1のタスク1-7として当初計画されていたが、既存事前学習モデルとの非互換のためM3-Bに延期された経緯がある（[milestones.md](../milestones.md) L87参照）。
>
> **milestones.md未修正箇所**: milestones.md L379のMVPセクションに `mel_fmin: 0.0 → 40.0` が「既存モデルとの互換性を完全に維持する」設定変更として残っているが、これは誤りである。mel_fmin変更は事前学習モデルとの互換性を破壊するため、MVPリストから削除すべき（M1チケットのセクション6でも同様の指摘あり）。

---

### 2-2. タスク3-7: アンチエイリアスフィルタ追加（2日）

**目的**: `ConvTranspose1d` によるアップサンプリング時に発生するエイリアシングノイズを、Kaiser窓sinc低域通過フィルタで除去する。

#### 2-2a. AntiAliasActivation クラスの実装

**新規追加先**: `infer/lib/infer_pack/modules.py`

```python
class AntiAliasActivation(nn.Module):
    """Kaiser窓sinc LPFによるアンチエイリアス処理
    BigVGAN / Vocos で標準的に採用されている手法。
    """
    def __init__(self, channels, upsample_rate, filt_size=12,
                 beta=14.769656459379492):
        super().__init__()
        self.filt_size = filt_size
        cutoff = 1.0 / upsample_rate
        kernel = self._design_kaiser_lpf(filt_size, cutoff, beta)
        self.register_buffer(
            'filt',
            kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1)
        )
        self.channels = channels

    @staticmethod
    def _design_kaiser_lpf(N, cutoff, beta):
        """Kaiser窓sinc低域通過フィルタの設計"""
        n = torch.arange(N) - (N - 1) / 2
        sinc = torch.sinc(2 * cutoff * n)
        window = torch.kaiser_window(N, periodic=False, beta=beta)
        kernel = sinc * window
        return kernel / kernel.sum()

    def forward(self, x):
        pad = self.filt_size // 2
        x = F.pad(x, (pad, pad), mode='reflect')
        return F.conv1d(x, self.filt, groups=self.channels)
```

#### 2-2b. GeneratorNSF への組み込み

**変更対象**: `infer/lib/infer_pack/models.py` L423-555 の `GeneratorNSF.__init__`

**挿入位置**: 各 `ConvTranspose1d`（`self.ups`）の直後、NSFソース加算およびResBlock処理の直前。

**現行のアップサンプル構成**:

| 設定 | 48kHz (`configs/v2/48k.json`) | 32kHz (`configs/v2/32k.json`) |
|------|------|------|
| `upsample_rates` | `[12, 10, 2, 2]` (積=480=hop_length) | `[10, 8, 2, 2]` (積=320=hop_length) |
| `upsample_kernel_sizes` | `[24, 20, 4, 4]` | `[20, 16, 4, 4]` |
| チャンネル推移 | 512 → 256 → 128 → 64 → 32 | 512 → 256 → 128 → 64 → 32 |

> **注意**: アンチエイリアスフィルタのカットオフ周波数は `1.0 / upsample_rate` で決まるため、48kHz版と32kHz版ではフィルタ特性が異なる。両方のconfigで正しく動作することをテストすること。

```python
# __init__ に追加
self.aa_filters = nn.ModuleList()
for i, u in enumerate(upsample_rates):
    ch = upsample_initial_channel // (2 ** (i + 1))
    self.aa_filters.append(AntiAliasActivation(ch, u))

# forward の変更
for i, (ups, noise_convs) in enumerate(zip(self.ups, self.noise_convs)):
    if i < self.num_upsamples:
        x = self.act_pre[i](x)
        x = ups(x)
        x = self.aa_filters[i](x)     # ← アンチエイリアスフィルタ挿入
        x_source = noise_convs(har_source)
        x = x + x_source
        # ... ResBlock処理 ...
```

**Generator (nono版) にも同様に追加**。

**推論速度への影響**: グループ畳み込み（`groups=channels`）のため追加計算コストは約-5%程度と軽微。フィルタ係数は `register_buffer` で保持するため学習パラメータは増加しない。

---

### 2-3. タスク3-8: MRD/CQTディスクリミネータ追加（2日）

**目的**: 既存の `MultiPeriodDiscriminatorV2`（MPD + DiscriminatorS）に加え、周波数方向のマルチスケール構造を評価するMRDおよびCQTディスクリミネータを学習ループに追加する。推論時は不要。

#### 2-3a. MultiResolutionDiscriminator (MRD) の実装

**新規追加先**: `infer/lib/infer_pack/models.py`

```python
class DiscriminatorR(nn.Module):
    """STFT-based single-resolution discriminator"""
    def __init__(self, resolution, channels=32, in_channels=1):
        super().__init__()
        self.resolution = resolution
        n_fft, hop_size, win_length = resolution
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(in_channels, channels, (3, 9), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 9),
                                  stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 9),
                                  stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 3), padding=(1, 1))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(channels, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        fmap = []
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

    @staticmethod
    def _stft(x, n_fft, hop_size, win_length):
        """STFT magnitude spectrogram"""
        window = torch.hann_window(win_length, device=x.device)
        x = x.squeeze(1)  # (B, T)
        stft = torch.stft(x, n_fft, hop_length=hop_size,
                          win_length=win_length, window=window,
                          return_complex=True)
        return stft.abs()  # (B, F, T)


class MultiResolutionDiscriminator(nn.Module):
    """UnivNet / BigVGAN 方式のマルチ解像度ディスクリミネータ"""
    def __init__(self, resolutions=None):
        super().__init__()
        if resolutions is None:
            resolutions = [
                (1024, 120, 600),   # 中解像度
                (2048, 240, 1200),  # 高解像度（低周波詳細）
                (512,  50,  240),   # 低解像度（高周波・過渡音）
            ]
        self.discriminators = nn.ModuleList([
            DiscriminatorR(r) for r in resolutions
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
```

#### 2-3b. CQTDiscriminator の実装

**新規追加先**: `infer/lib/infer_pack/models.py`

CQT（Constant-Q Transform）は等比間隔で周波数を分解するため、倍音列（基本周波数の整数倍）を自然に捉えられる。歌声のビブラートや声区遷移の評価に適する。

```python
class CQTSubDiscriminator(nn.Module):
    """CQTベースの単一ディスクリミネータ。
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
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, channels, (3, 9), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 9),
                                  stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 9),
                                  stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 3), padding=(1, 1))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(channels, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        fmap = []
        cqt_spec = self._compute_cqt(x)
        x = cqt_spec.unsqueeze(1)
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class CQTDiscriminator(nn.Module):
    """CQTディスクリミネータのラッパー。
    MultiPeriodDiscriminatorV2.forward(y, y_hat) と同じインターフェース
    (y_d_rs, y_d_gs, fmap_rs, fmap_gs) を返す。
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.disc = CQTSubDiscriminator(**kwargs)

    def forward(self, y, y_hat):
        y_d_r, fmap_r = self.disc(y)
        y_d_g, fmap_g = self.disc(y_hat)
        return [y_d_r], [y_d_g], [fmap_r], [fmap_g]
```

> **重要**: `CQTDiscriminator.forward(y, y_hat)` は `MultiPeriodDiscriminatorV2.forward` と同じ4値タプル `(y_d_rs, y_d_gs, fmap_rs, fmap_gs)` を返すラッパーとして実装する。これにより学習ループの `discriminator_loss` / `generator_loss` / `feature_loss` をそのまま再利用できる。`MultiResolutionDiscriminator` は既にこのインターフェースを満たしている。

**依存パッケージ**: CQT計算には `nnAudio` の使用を検討（GPU上で効率的なCQT計算が可能）。`pyproject.toml` に `nnAudio>=0.3.2` を追加する必要がある。代替として `torchaudio.transforms.KCQT` も選択肢。

> **実装補足**: 上記コードの `self._compute_cqt(x)` メソッドは未定義。実装時に `nnAudio.features.CQT` または `torchaudio.transforms.KCQT` を使って実装する必要がある。選択するライブラリによって `__init__` でのCQT変換器の初期化方法が異なるため、実装時に確定する。

#### 2-3c. 学習ループへの統合

**変更対象**: `infer/modules/train/train.py`

**現行のディスクリミネータ処理** (L55-58, L171, L451-468):
```python
# L55-58: インポート
from infer.lib.infer_pack.models import (
    SynthesizerTrnMs768NSFsid as RVC_Model_f0,
    SynthesizerTrnMs768NSFsid_nono as RVC_Model_nof0,
    MultiPeriodDiscriminatorV2 as MultiPeriodDiscriminator,
)

# L171: インスタンス化
net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)

# L451: Discriminator学習ステップ
y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)

# L462-468: Generator学習ステップ
y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
loss_fm = feature_loss(fmap_r, fmap_g)
loss_gen, losses_gen = generator_loss(y_d_hat_g)
loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
```

> **注意**: train.pyのインポート (L55-58) では `MultiPeriodDiscriminatorV2` を `MultiPeriodDiscriminator` にリネームしてインポートしている。`MultiPeriodDiscriminatorV2` は periods `[2,3,5,7,11,17,23,37]` + `DiscriminatorS` の構成。

**変更後**:
```python
# インポートに追加
from infer.lib.infer_pack.models import (
    MultiResolutionDiscriminator,
    CQTDiscriminator,
)

# L171付近にMRD/CQTのインスタンス化を追加
net_d_mrd = MultiResolutionDiscriminator().cuda(rank)
net_d_cqt = CQTDiscriminator(sample_rate=hps.data.sampling_rate).cuda(rank)
net_d_mrd = DDP(net_d_mrd, device_ids=[rank])
net_d_cqt = DDP(net_d_cqt, device_ids=[rank])
optim_d_mrd = torch.optim.AdamW(net_d_mrd.parameters(), ...)
optim_d_cqt = torch.optim.AdamW(net_d_cqt.parameters(), ...)

# Discriminator学習ステップ（既存のMPDに並列追加）
y_d_hat_r_mrd, y_d_hat_g_mrd, _, _ = net_d_mrd(wave, y_hat.detach())
loss_disc_mrd, _, _ = discriminator_loss(y_d_hat_r_mrd, y_d_hat_g_mrd)
y_d_hat_r_cqt, y_d_hat_g_cqt, _, _ = net_d_cqt(wave, y_hat.detach())
loss_disc_cqt, _, _ = discriminator_loss(y_d_hat_r_cqt, y_d_hat_g_cqt)
c_mrd = getattr(hps.train, 'c_disc_mrd', 1.0)
c_cqt = getattr(hps.train, 'c_disc_cqt', 0.5)
loss_disc_total = loss_disc + c_mrd * loss_disc_mrd + c_cqt * loss_disc_cqt

# Generator学習ステップ（feature matching lossも追加）
y_d_hat_r_mrd, y_d_hat_g_mrd, fmap_r_mrd, fmap_g_mrd = net_d_mrd(wave, y_hat)
y_d_hat_r_cqt, y_d_hat_g_cqt, fmap_r_cqt, fmap_g_cqt = net_d_cqt(wave, y_hat)
loss_fm_mrd = feature_loss(fmap_r_mrd, fmap_g_mrd)
loss_fm_cqt = feature_loss(fmap_r_cqt, fmap_g_cqt)
loss_gen_mrd, _ = generator_loss(y_d_hat_g_mrd)
loss_gen_cqt, _ = generator_loss(y_d_hat_g_cqt)
loss_gen_all = (loss_gen + loss_fm + loss_mel + loss_kl
                + c_mrd * (loss_gen_mrd + loss_fm_mrd)
                + c_cqt * (loss_gen_cqt + loss_fm_cqt))
```

> **注意**: `getattr(hps.train, 'c_disc_mrd', 1.0)` を使用し、旧configファイル（`c_disc_mrd` が未定義）でもデフォルト値で動作するようにする。これによりR6（設定ファイルの後方互換）要件を満たす。

**TensorBoardロギングの更新** (L491-508):
- `loss/d_mrd/total`、`loss/d_cqt/total` をスカラー辞書に追加
- `loss/g/fm_mrd`、`loss/g/fm_cqt` を追加

**チェックポイント保存の更新** (L523-553):
- MRD/CQTのチェックポイントも保存（`D_mrd_*.pth`、`D_cqt_*.pth`）
- ただし推論時に不要のため、`process_ckpt.py` の `savee()` には含めない

**重要**: MRD/CQTは学習時のみ使用し、推論コードに変更は不要。ユーザーのファインチューニング時にも自動的にMRD/CQTが有効になる。

#### 2-3d. 損失重みのバランス調整

3つのディスクリミネータ（MPD + MRD + CQT）の損失を同時使用する場合、バランス調整が必要。設定ファイルにハイパーパラメータとして追加する。

**`configs/v2/48k.json` の `train` セクションに追加**:
```json
{
  "train": {
    "c_disc_mrd": 1.0,
    "c_disc_cqt": 0.5,
    ...
  }
}
```

初期値はBigVGAN論文の推奨値を参考に設定し、事前学習時に調整する。

---

### 2-4. タスク3-9: 事前学習再実行（1日作業 + GPU 14-30h）

**目的**: タスク3-6〜3-8の全変更を反映した新しい事前学習済みモデルを生成する。

**実行環境**: RTX 4090 24GB（ローカル）またはCloud GPU（A100推奨）

**事前学習の入力**:
- M2で準備した日本語歌声データセット（kushinada HuBERT特徴量）
- 変更後の設定ファイル（`mel_fmin=40.0`、SnakeBeta関連パラメータ）
- M3-Aで追加した損失関数（Multi-Resolution STFT Loss等）

**事前学習の構成**:
- Generator: SnakeBeta + アンチエイリアスフィルタ付き GeneratorNSF
- Discriminator: MPD + MRD + CQT
- SSLモデル: kushinada-hubert-base（M2-Aで統合済み）
- mel_fmin: 40.0

**出力成果物**:
- `assets/pretrained_v2/f0G48k_snakebeta.pth` - Generator事前学習済みモデル
- `assets/pretrained_v2/f0D48k_snakebeta.pth` - Discriminator事前学習済みモデル（MPD部分）
- `assets/pretrained_v2/f0G32k_snakebeta.pth` - 32kHz版Generator
- `assets/pretrained_v2/f0D32k_snakebeta.pth` - 32kHz版Discriminator

**注意**:
- M2で作成した事前学習モデルはSnakeBeta導入により再利用不可
- MRD/CQTのチェックポイントは事前学習成果物として同梱するが、推論には不要
- GPU所要時間の見積もり: RTX 4090で約14-20h、A100で約8-12h、V100で約24-30h

---

## 3. エージェントチーム構成

### Phase 3-B 実装担当（Week 7-8）

| 役割 | 担当範囲 | 必要スキル |
|------|---------|-----------|
| **実装リード** | SnakeBeta、アンチエイリアスフィルタの設計・実装・テスト | PyTorch nn.Module設計、信号処理（DSP）の基礎 |
| **学習パイプライン担当** | MRD/CQTディスクリミネータの実装、学習ループ統合、損失バランス調整 | GAN学習の経験、分散学習（DDP） |
| **事前学習オペレーター** | GPU環境準備、事前学習実行・監視、チェックポイント管理 | Cloud GPU運用、TensorBoardモニタリング |
| **評価担当** | Go/No-Go判定③の評価実行（MCD/CER/MOS） | M0の評価パイプライン運用 |

### 依存関係

```
タスク3-6 (SnakeBeta + mel_fmin)
    │
    ├──→ タスク3-7 (アンチエイリアス)    ← 3-6と並行可能だが、同一ファイル改修のため直列推奨
    │         │
    │         v
    └────→ タスク3-9 (事前学習再実行)   ← 3-6 + 3-7 完了後に実行
              │
    タスク3-8 (MRD/CQT)               ← 3-6/3-7と独立に並行実装可能
              │                         ただし事前学習には3-8も含める
              v
         Go/No-Go判定③
```

---

## 4. 提供範囲・テスト項目

### 4-1. ユニットテスト

| # | テスト項目 | 検証内容 |
|---|-----------|---------|
| U1 | SnakeBeta forward | 入力テンソル `(B=2, C=128, T=100)` に対して出力形状が同一であること。`alpha`, `beta` パラメータが勾配を持つこと |
| U2 | SnakeBeta 数値安定性 | `alpha`, `beta` が極端な値（0.01, 100.0）でも NaN/Inf が発生しないこと |
| U3 | AntiAliasActivation | Kaiser窓LPFのフィルタ係数合計が1.0であること。入出力テンソル長が同一であること |
| U4 | DiscriminatorR forward | 入力 `(B=2, 1, T=48000)` に対して出力スカラーとfeature mapリストが返ること |
| U5 | CQTDiscriminator forward | 入力 `(B=2, 1, T=48000)` に対して `CQTDiscriminator.forward(y, y_hat)` が4値タプル `(y_d_rs, y_d_gs, fmap_rs, fmap_gs)` を返すこと |
| U6 | ResBlock1 + SnakeBeta | 改修後のResBlock1がjit.script互換を維持すること（`__prepare_scriptable__` が動作すること） |
| U7 | ResBlock2 + SnakeBeta | 改修後のResBlock2が正常動作すること。SnakeBetaインスタンス数がdilation数（2個）と一致すること |
| U8 | Generator (nono) + SnakeBeta | F0なしモデル用Generatorの `forward` がSnakeBeta置換後に動作すること |
| U9 | mel_fmin=40.0でのmel生成 | `mel_processing.py` の `spec_to_mel_torch` / `mel_spectrogram_torch` が `fmin=40.0` で正常にmelスペクトログラムを生成できること |

### 4-2. 統合テスト

| # | テスト項目 | 検証内容 |
|---|-----------|---------|
| I1 | GeneratorNSF forward | SnakeBeta + アンチエイリアス付きGeneratorNSFで、入力 `(B=1, 192, T=100)` + F0テンソルに対して波形出力 `(B=1, 1, T*480)` が得られること |
| I2 | SynthesizerTrnMs768NSFsid 統合 | 改修後のGeneratorNSFを `self.dec` として持つSynthesizerがforward/inferの両方で動作すること |
| I3 | SynthesizerTrnMs768NSFsid_nono 統合 | F0なしモデル（`Generator` クラス）もSnakeBeta対応後に動作すること |
| I4 | 学習ループ1ステップ | `train.py` の学習ループで、MPD + MRD + CQTの3ディスクリミネータを含む1ステップが正常完了すること（VRAM < 16GB on RTX 4070 Ti Super、batch_size=2） |
| I5 | チェックポイント保存/復帰 | 新しいGeneratorのstate_dictが保存/ロードできること。新旧チェックポイントの誤ロード時にエラーメッセージが出ること |
| I6 | 32kHz config互換 | `configs/v2/32k.json` の `upsample_rates=[10,8,2,2]` でもGeneratorNSF/Generatorが正常動作すること。アンチエイリアスフィルタのカットオフが各レートに対して正しく設定されること |
| I7 | 旧config後方互換 | `c_disc_mrd` / `c_disc_cqt` が未定義の旧configファイルで `train.py` がデフォルト値で動作すること |

### 4-3. 品質評価テスト

| # | テスト項目 | 検証内容 | 判定基準 |
|---|-----------|---------|---------|
| Q1 | MCD（Mel Cepstral Distortion） | 事前学習後モデルでのMCD値 | < 6.5 dB |
| Q2 | CER（Character Error Rate） | Whisperによる歌詞認識精度 | < 12% |
| Q3 | MOS（Mean Opinion Score） | 主観評価スコア | > 3.8 (Go) / > 3.5 (M3完了) |
| Q4 | 高音域アーティファクト | C5以上の女声・ファルセット変換 | 金属的ノイズの消失/軽減 |
| Q5 | 声区遷移 | 胸声→ヘッドボイス遷移の自然さ | 急変の緩和 |
| Q6 | A/B比較テスト | M2ベースライン vs M3-B成果物 | M3-Bが有意に優位 |

### 4-4. VRAM・パフォーマンステスト

| # | テスト項目 | 環境 | 判定基準 |
|---|-----------|------|---------|
| P1 | 推論時VRAM | RTX 4070 Ti Super 16GB | < 4 GB（SnakeBeta + アンチエイリアス追加後） |
| P2 | 学習時VRAM | RTX 4090 24GB | < 20 GB（MPD + MRD + CQT、batch_size=4） |
| P3 | 推論速度 | RTX 4070 Ti Super | 現行比 -10% 以内（アンチエイリアスフィルタ分の許容劣化） |
| P4 | リアルタイム変換レイテンシ | RTX 4070 Ti Super | `tools/rvc_for_realtime.py` で実用的なレイテンシを維持 |

---

## 5. 懸念事項とレビュー項目

### 5-1. 技術的懸念

| # | 懸念事項 | リスク | 対策 |
|---|---------|--------|------|
| C1 | **SnakeBetaの数値安定性**: `alpha.exp()` が大きくなると `sin(alpha * x)` の振動が極端になり、学習不安定を引き起こす可能性 | 中 | `alpha_logscale=True` で対数スケール管理。学習初期のalpha値をTensorBoardで監視。必要に応じてクランプ（`alpha.clamp(min=-5, max=5)`）を追加 |
| C2 | **3ディスクリミネータの損失バランス**: MPD + MRD + CQTの損失重みが不適切だとGenerator学習崩壊のリスク | 高 | 各ディスクリミネータの損失を個別にTensorBoardにログ。初期重み `c_disc_mrd=1.0, c_disc_cqt=0.5` から開始し、事前学習の最初の1000ステップで損失スケールを観察して調整 |
| C3 | **CQTライブラリ依存**: `nnAudio` のGPUメモリ使用量がバッチ処理時に大きい可能性 | 中 | まずMRDのみで事前学習を開始し、CQTは後から追加実験として検証。CQT抜きでも効果が十分であればスキップ可能 |
| C4 | **mel_fmin変更とNSFの相互作用**: mel_fmin=40.0にすると低音域の情報が失われ、低音の男声変換品質に影響する可能性 | 低 | 歌声の基本周波数は通常80Hz以上。40Hz以下にはDCオフセットとノイズしか含まれないため実質的な影響はない。OpenVPI（DiffSinger等）で実績あり |
| C5 | **jit.script互換性**: SnakeBetaのパラメータ化された活性化関数が `torch.jit.script()` と互換か | 中 | `__prepare_scriptable__` メソッドを適切に実装。jit.scriptテストをU6で検証 |
| C6 | **事前学習再実行の時間的リスク**: GPU 14-30hの事前学習中にエラーが発生するとWeek 8のスケジュールに影響 | 中 | 1000ステップ目でサニティチェック（loss値の異常検知）。チェックポイント保存間隔を短く設定（500ステップ毎） |

### 5-2. コードレビュー重点項目

| # | レビュー項目 | 確認ポイント |
|---|------------|------------|
| R1 | `SnakeBeta.forward` の勾配フロー | `alpha`, `beta` パラメータが正しく勾配を受け取ること。`torch.sin()` の勾配が数値的に安定すること |
| R2 | `AntiAliasActivation` のフィルタ設計 | Kaiser窓のβ値、カットオフ周波数がアップサンプル率に対して適切であること |
| R3 | `GeneratorNSF.remove_weight_norm()` | SnakeBeta導入後も `remove_weight_norm()` と `__prepare_scriptable__()` が正常動作すること（推論最適化に必要） |
| R4 | MRD/CQTの `forward` シグネチャ | 既存の `MultiPeriodDiscriminatorV2.forward(y, y_hat)` と同じインターフェース（`y_d_rs, y_d_gs, fmap_rs, fmap_gs` を返す）を満たすこと |
| R5 | 学習ループのoptimizer管理 | MRD/CQT用のoptimizerが正しく初期化・ステップ・チェックポイント管理されていること。学習再開時にMRD/CQTのoptimizerも正しく復元されること |
| R6 | 設定ファイルの後方互換 | `mel_fmin=40.0` や `c_disc_mrd` 等の新設定がない旧configファイルでも `train.py` がデフォルト値で動作すること |

### 5-3. ロールバック計画

- **事前学習前**: gitタグ `pre-m3b` を作成。SnakeBeta導入前のコードはgitタグで保存
- **事前学習後に品質不足**: M2のチェックポイントに戻し、SnakeBetaなしの現行アーキテクチャで継続
- **一部コンポーネントの問題**: MRD/CQTは学習時のみのため無効化が容易。SnakeBeta/アンチエイリアスは事前学習再実行が必要なためロールバックコストが高い

---

## 6. 一から作り直すとしたら（M3フェーズ全体の理想設計）

M3フェーズ（Phase 3-A 損失関数 + Phase 3-B ボコーダ改善）を白紙から設計し直す場合、以下のアプローチを取る。

### 6-1. 統合的なボコーダ近代化

現在の計画ではSnakeBeta・アンチエイリアス・MRD/CQTを段階的に導入するが、これらは本来BigVGANのAMPBlock（Anti-aliased Multi-Periodicity Block）として統合設計されたものである。理想的には個別導入ではなく、**AMPBlockをまるごと移植**し、ResBlock1/ResBlock2を完全に置換するアプローチが望ましい。

```python
class AMPBlock(nn.Module):
    """Anti-aliased Multi-Periodicity Block (BigVGAN)
    SnakeBeta + 畳み込み + AntiAlias を一体化。
    """
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.activation_pre = SnakeBeta(channels)
        self.conv1 = weight_norm(Conv1d(...))
        self.aa_filter1 = AntiAliasActivation(channels, upsample_rate=1)
        self.activation_mid = SnakeBeta(channels)
        self.conv2 = weight_norm(Conv1d(...))
        self.aa_filter2 = AntiAliasActivation(channels, upsample_rate=1)
```

これにより、SnakeBetaとアンチエイリアスフィルタが常にペアで動作することが保証され、テストの分離が容易になる。

### 6-2. ディスクリミネータの統合管理

MPD・MRD・CQTを個別のoptimizerで管理するのではなく、**CompositeDiscriminator**として統合し、単一のoptimizerで管理する設計が理想。

```python
class CompositeDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.mpd = MultiPeriodDiscriminatorV2()
        self.mrd = MultiResolutionDiscriminator()
        self.cqt = CQTDiscriminator()

    def forward(self, y, y_hat):
        # 全ディスクリミネータの出力をマージして返す
```

### 6-3. mel_fmin変更のタイミング

mel_fmin変更はM1で実施し、以降の全マイルストーンで一貫して `mel_fmin=40.0` を使用すべきだった。M3-Bまで延期したことで、M2の事前学習成果物が再利用不可になるコストが発生している。理想的にはM0の段階で `configs/v2/48k_singing.json` を新規作成し、既存configとの分離を最初から行うべきだった。

### 6-4. iSTFTベースへの移行検討

M3-Bの改善（SnakeBeta + アンチエイリアス）は現行HiFi-GAN構造の延命措置であり、根本的にはiSTFTベースのデコーダ（Vocos / HiFTNet）への移行が長期的に望ましい。理想設計では、M3でHiFTNetへの移行を一括で行い、SnakeBeta/アンチエイリアスの個別導入ステップをスキップする。ただし、HiFTNetとNSFの統合は未解決の研究課題を含むため、リスク管理の観点から現在の段階的アプローチは妥当。

---

## 7. 後続タスクへの連絡事項

### M4（高度な最適化）への引き継ぎ

1. **事前学習チェックポイントの命名規則**: M3-Bで作成する事前学習モデルは `f0G48k_snakebeta.pth` の命名とする。M4でさらなるアーキテクチャ変更を行う場合は、この命名規則を拡張すること（例: `f0G48k_snakebeta_hiftnet.pth`）

2. **Go/No-Go判定③の結果伝達**: Week 8終了時のMCD/CER/MOSの定量値と、高音域アーティファクト・声区遷移の主観評価結果をM4のキックオフ資料として引き渡す

3. **MRD/CQTの効果分析**: 事前学習時のTensorBoardログから、MRD/CQTそれぞれの寄与度を分析してM4に報告。CQTの効果が限定的であれば、M4ではCQTを除外してVRAM確保に充てることを検討

4. **mel_fmin=40.0の影響確認**: 低音域（C2-C3）の男声変換品質に劣化がないことを確認し、結果をM4に報告。劣化がある場合は `mel_fmin=20.0` へのフォールバック検討をM4で実施

5. **SnakeBeta alpha/beta パラメータの学習済み値**: 事前学習後の各ResBlockにおける `alpha`, `beta` の収束値を記録。M4でアーキテクチャ変更（HiFTNet等）を行う際の初期値設計の参考とする

6. **コード変更の影響範囲まとめ**:
   - **推論に影響するファイル**: `infer/lib/infer_pack/modules.py`（SnakeBeta, AntiAlias追加）、`infer/lib/infer_pack/models.py`（GeneratorNSF, Generator改修）、`configs/v2/*.json`（mel_fmin変更）
   - **学習のみに影響するファイル**: `infer/modules/train/train.py`（MRD/CQT統合）、`infer/lib/infer_pack/models.py`（MRD/CQTクラス追加）
   - **推論に影響しないファイル**: MRD/CQTディスクリミネータのチェックポイントは推論時に不要

### WebUI・ユーザー向けの連絡事項

7. **既存モデルとの非互換**: M3-B以降の事前学習モデルで学習したユーザーモデルは、M3-B以前のRVCでは使用不可。逆も同様。WebUIにモデルバージョン検知と警告表示の実装が必要（M4で対応を検討）

8. **`infer-web.py` の変更なし**: M3-BではWebUI側の変更は不要。SnakeBeta/アンチエイリアスはGeneratorの内部変更であり、推論パイプライン（`pipeline.py`）やWebUIのインターフェースには影響しない

9. **リアルタイム変換への影響**: `tools/rvc_for_realtime.py` および `infer/lib/rtrvc.py` はGeneratorの `forward` を呼び出すのみのため、SnakeBeta/アンチエイリアスの導入は透過的。ただし推論速度が約5%低下する可能性があるため、レイテンシ測定を実施すること
