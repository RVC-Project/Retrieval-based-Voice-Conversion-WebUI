# M1: 即効性改善（既存モデル互換維持）

## メタ情報
- **マイルストーン**: M1
- **フェーズ**: Week 1-2（M0と並列実行）
- **工数見積もり**: 10人日
- **GPU要件**: RTX 4070 Ti Super 16GB（開発・検証）、RTX 4090 24GB（本格学習）
- **前提タスク**: なし（M0と独立して実行可能）
- **ステータス**: 実装完了（評価実行除く）
- **関連マイルストーン**: [milestones.md](../milestones.md) > M1セクション
- **追加パッケージ**: `auraloss>=0.4.0`（pyproject.toml に追記）

### 実装結果サマリー（レビュー修正含む）

| 項目 | 当初仕様 | 実装結果 |
|------|---------|---------|
| Discriminator weight_decay | 0.01（G/D同一） | **G=0.01, D=0**（レビューで修正。Dに正則化不要） |
| c_mrstft | 2.5 | **5.0**（レビューで引き上げ） |
| MRSTFT hop_sizes | (120, 240, 50) | **(256, 512, 128)**（レビューで修正） |
| MRSTFT win_lengths | (600, 1200, 240) | **(1024, 2048, 512)**（レビューで修正） |
| 演歌 f0_max | 1100 | **900**（レビューで修正） |
| アニソン f0_max | 1400 | **1200**（レビューで修正） |
| segment_size (32k) | 12800→25600 | **25600**（実装通り） |
| p_dropout | 0.1 | **0.1**（実装通り） |
| y_hat_mel amp_dtype | 未考慮 | **バグ修正済み**（y_hat_mel.to(amp_dtype)） |

---

## 1. タスク目的とゴール

### 目的
日本語歌声変換の品質を、**既存の事前学習済みモデルとの互換性を完全に維持したまま**向上させる。具体的には以下の3軸で改善を狙う。

1. **F0推定精度の向上** --- FCPE統合・F0レンジ拡張・filter_radiusデフォルト変更により、歌声特有の広い音域やビブラートを正確に捉える
2. **学習品質の向上** --- Dropout/Weight Decay追加、segment_size拡張、Multi-Resolution STFT損失追加、bfloat16移行により、少量データからの汎化性能を改善する
3. **ユーザビリティ向上** --- 歌声プリセット追加により、初心者でもワンクリックで最適パラメータを適用可能にする

### 成功条件（Go/No-Go判定 Week 2終了時）

| 基準 | Go条件 | No-Go時の対応 |
|------|--------|--------------|
| MCD (Mel Cepstral Distortion) | ベースラインから5%以上改善 | パラメータ再調整（segment_size, filter_radius検証） |
| F0 RMSE | ベースラインから10%以上改善 | FCPEパラメータ微調整、f0_min/f0_max変更 |
| リアルタイムレイテンシ | 200ms以下を維持 | segment_size縮小で対応 |
| 既存モデル互換性 | 完全維持（既存.pthファイルがそのまま読み込め、推論結果が変わらない） | 変更箇所をリバート |

### 既存モデル互換性の定義
- `assets/pretrained_v2/` 配下の事前学習済みモデル（G_*.pth, D_*.pth）がそのまま使える
- ユーザが既に学習済みの `assets/weights/*.pth` がそのまま推論に使える
- モデルアーキテクチャ（`SynthesizerTrnMs768NSFsid`, `SynthesizerTrnMs768NSFsid_nono`）の入出力インタフェースに変更なし
- 設定JSON（`configs/v2/*.json`）の変更は**新規学習時のみ影響**し、既存モデルの推論には影響しない

---

## 2. 実装する内容の詳細

### サブタスク一覧

- [x] **1-1**: FCPE統合（メインパイプライン） --- 実装完了（pipeline.py, extract_f0_print.py）
- [x] **1-2**: F0レンジ拡張（65-1400Hz） --- 実装完了
- [x] **1-3**: filter_radius デフォルト変更（3→1） --- 実装完了
- [x] **1-4**: Dropout(0.1) + Weight Decay(Generator=0.01, Discriminator=0) --- 実装完了（レビューでDiscriminator weight_decay=0に修正）
- [x] **1-5**: segment_size拡張（48k: 34560, 32k: 25600） --- 実装完了
- [x] **1-6**: 歌唱向け前処理パラメータ --- 実装完了
- [ ] ~~**1-7**: mel_fmin変更（0→40Hz）~~ --- **M3-Bに延期**（互換性リスク）
- [x] **1-8**: Multi-Resolution STFT損失追加 --- 実装完了（c_mrstft=5.0, fft_sizes=(1024,2048,512), hop_sizes=(256,512,128), win_lengths=(1024,2048,512)）
- [x] **1-9**: 歌声プリセット（WebUI） --- 実装完了（f0_presets.py: J-POP, 演歌(f0_max=900), アニソン(f0_max=1200,fcpe), 話し声）
- [x] **1-10**: bfloat16/fp16混合精度対応 --- 実装完了（train.py）
- [ ] **1-11**: 評価実行 + ベースライン比較 --- 未実施（実際の音声データが必要）

---

### 1-1: FCPE統合（メインパイプライン）
**工数**: 1日 | **変更ファイル**: `infer/modules/vc/pipeline.py`, `infer/modules/train/extract/extract_f0_print.py`

#### 背景
現在の推論パイプラインでは `pm`, `harvest`, `crepe`, `rmvpe` の4つのF0抽出メソッドを提供しているが、FCPEは非対応。FCPEはtorchfcpeパッケージ（既にpyproject.tomlに`torchfcpe>=0.0.4`として依存追加済み）を使い、リアルタイム処理に適した高速かつ高精度なF0推定が可能。

#### 変更内容

**`infer/modules/vc/pipeline.py` -- `get_f0` メソッド（L75-163）**

`rmvpe` の分岐（L131-146）の後に `fcpe` 分岐を追加:

```python
elif f0_method == "fcpe":
    if not hasattr(self, "model_fcpe"):
        from torchfcpe import spawn_bundled_infer_model
        logger.info("Loading FCPE model")
        self.model_fcpe = spawn_bundled_infer_model(device=self.device)
    audio_tensor = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(-1).to(self.device)
    f0 = self.model_fcpe.infer(
        audio_tensor,
        sr=self.sr,
        decoder_mode="local_argmax",
        threshold=0.006,
    )
    f0 = f0.squeeze().cpu().numpy()
```

**`infer/modules/train/extract/extract_f0_print.py` -- `compute_f0` メソッド（L44-89）**

`rmvpe` の分岐（L82-88）の後に `fcpe` 分岐を追加:

> **注意**: 現在の `extract_f0_print.py` は `torch` をインポートしていない。FCPE分岐では `torch.from_numpy()` を使うため、ファイル先頭に `import torch` を追加する必要がある。

```python
elif f0_method == "fcpe":
    import torch  # extract_f0_print.pyには未インポートのため必要
    if not hasattr(self, "model_fcpe"):
        from torchfcpe import spawn_bundled_infer_model
        print("Loading FCPE model")
        self.model_fcpe = spawn_bundled_infer_model(device="cpu")
    audio_tensor = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(-1)
    f0 = self.model_fcpe.infer(
        audio_tensor,
        sr=self.fs,
        decoder_mode="local_argmax",
        threshold=0.006,
    )
    f0 = f0.squeeze().cpu().numpy()
```

**`infer-web.py` -- F0メソッドのラジオボタン**

- `f0method0`（L777: 単独推論タブ）: `choices` に `"fcpe"` を追加。現在は `["pm", "harvest", "crepe", "rmvpe"]`（DML時は `["pm", "harvest", "rmvpe"]`）
- `f0method1`（L899: バッチ推論タブ）: 同上
- `f0method8`（L1154: 学習時F0抽出タブ）: `choices` に `"fcpe"` を追加。現在は `["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"]`（推論用とは選択肢が異なる点に注意）

#### 注意点
- FCPEモデルはtorchfcpe内にバンドルされており、追加のモデルファイルダウンロードは不要
- `decoder_mode="local_argmax"` はリアルタイム向け。バッチ推論時は `"local_argmax"` でも十分だが、オフライン高精度が必要なら `"argmax"` も検討可能
- `threshold=0.006` はデフォルト値。ブレスが多い曲では `0.003` に下げることも有効（プリセットで対応）
- `is_half` 対応：FCPEモデル自体がfloat32前提で動作するため、half精度変換は不要（内部でfloat32処理）

---

### 1-2: F0レンジ拡張（65-1400Hz）
**工数**: 0.5日 | **変更ファイル**: `infer/modules/vc/pipeline.py`, `infer/modules/train/extract/extract_f0_print.py`

#### 背景
現在の F0 レンジ:
- `pipeline.py` L87-88: `f0_min = 50`, `f0_max = 1100`
- `extract_f0_print.py` L39-40: `self.f0_max = 1100.0`, `self.f0_min = 50.0`

日本語歌唱の音域を考慮すると以下が適切:
- 女性ソプラノ: C3(131Hz) - C6(1047Hz)、ファルセットでE6(1319Hz)まで
- 男性テナー: C2(65Hz) - C5(523Hz)
- アニソン高音域: F6(1397Hz)程度まで

#### 変更内容

**`infer/modules/vc/pipeline.py` L87-88**
```python
# 変更前
f0_min = 50
f0_max = 1100

# 変更後
f0_min = 65
f0_max = 1400
```

**`infer/modules/train/extract/extract_f0_print.py` L39-40**
```python
# 変更前
self.f0_max = 1100.0
self.f0_min = 50.0

# 変更後
self.f0_max = 1400.0
self.f0_min = 65.0
```

#### 注意点
- F0メルスケール変換のmin/max（L89-90, L41-42）も自動的に追従する（計算値を使っているため）
- 50Hz以下の基本周波数は音声信号ではほぼ存在せず、ノイズ誤検出の原因になるため65Hzに引き上げ
- 1400Hzへの上限拡張により、ファルセットやホイッスルボイスの一部も追跡可能になる
- `f0_coarse` の量子化（256段階）の分解能がわずかに低下するが、実用上問題ない

---

### 1-3: filter_radius デフォルト変更（3→1）
**工数**: 0.5日 | **変更ファイル**: `infer-web.py`, `tools/infer_cli.py`, `tools/infer_batch_rvc.py`, `tools/app.py`

#### 背景
`filter_radius` はharvestの結果に対するメディアンフィルタの半径。現在のデフォルト値 `3` は、ビブラートを過度に平滑化してしまう問題がある。`pipeline.py` L108では `filter_radius > 2` の場合のみ `signal.medfilt(f0, 3)` が適用される。デフォルトを `1` にすることで、harvestメソッド使用時もビブラートが保存される。

#### 変更内容

**`infer-web.py` -- 2箇所**
- L822: `filter_radius0` の `value=3` → `value=1`
- L960: `filter_radius1` の `value=3` → `value=1`

**`tools/infer_cli.py` L30**
```python
# 変更前
parser.add_argument("--filter_radius", type=int, default=3, help="filter radius")

# 変更後
parser.add_argument("--filter_radius", type=int, default=1, help="filter radius")
```

**`tools/infer_batch_rvc.py` L30**
```python
# 同上の変更
```

**`tools/app.py` L72-78 付近**
- `filter_radius0` の `value=3` → `value=1`

#### 注意点
- APIエンドポイント (`api_240604.py`) には `filter_radius` パラメータは存在しないことを確認済み（変更不要）
- 既存ユーザへの影響：デフォルト値の変更のみで、ユーザが明示的に `filter_radius=3` を指定していた場合はそのまま動作する
- `filter_radius=0` ではフィルタリングなし。演歌のこぶし表現を完全に保存したい場合に有効

---

### 1-4: Dropout(0.1) + Weight Decay(Generator=0.01, Discriminator=0)
**工数**: 0.5日 | **変更ファイル**: `configs/v2/48k.json`, `configs/v2/32k.json`, `infer/modules/train/train.py`

#### 背景
現在の設定:
- `p_dropout`: `0`（`configs/v2/48k.json` L35, `configs/v2/32k.json` L35）
- Weight Decay: なし（AdamW のデフォルト `weight_decay=0` 相当）

少量データ（10分程度）での学習では過学習が問題になりやすい。Dropoutとweight decayの導入で汎化性能を改善する。

#### 変更内容

**`configs/v2/48k.json` -- model セクション**
```json
"p_dropout": 0.1
```
（L35: `"p_dropout": 0` → `"p_dropout": 0.1`）

**`configs/v2/32k.json` -- 同上**

**`infer/modules/train/train.py` -- AdamW初期化（L174-185）**
```python
# 変更前
optim_g = torch.optim.AdamW(
    net_g.parameters(),
    hps.train.learning_rate,
    betas=hps.train.betas,
    eps=hps.train.eps,
)
optim_d = torch.optim.AdamW(
    net_d.parameters(),
    hps.train.learning_rate,
    betas=hps.train.betas,
    eps=hps.train.eps,
)

# 変更後（レビュー修正: Discriminator weight_decay=0）
optim_g = torch.optim.AdamW(
    net_g.parameters(),
    hps.train.learning_rate,
    betas=hps.train.betas,
    eps=hps.train.eps,
    weight_decay=getattr(hps.train, 'weight_decay', 0.01),
)
optim_d = torch.optim.AdamW(
    net_d.parameters(),
    hps.train.learning_rate,
    betas=hps.train.betas,
    eps=hps.train.eps,
    weight_decay=0,  # Discriminatorにはweight decayを適用しない
)
```

**`configs/v2/48k.json` -- train セクション**に追加:
```json
"weight_decay": 0.01
```

**`configs/v2/32k.json` -- 同上**

#### 互換性への影響
- `p_dropout` は `TextEncoder` の初期化パラメータ（`models.py` L40: `self.p_dropout = float(p_dropout)`）として使われる。既存の学習済みモデルは `p_dropout=0` で学習されているが、**推論時にはDropoutは無効化される**（`model.eval()` で自動的にDropout無効化）ため互換性に影響なし
- `weight_decay` は学習プロセスのみに影響し、推論には関係しない
- `getattr` のフォールバックにより、旧JSONファイルでも動作する

---

### 1-5: segment_size拡張（17280→34560 / 12800→25600）
**工数**: 0.5日 | **変更ファイル**: `configs/v2/48k.json`, `configs/v2/32k.json`

#### 背景
`segment_size` は学習時にランダムクロップされるオーディオセグメントの長さ（サンプル数）。
- 48kHz: 現在 17280サンプル = 0.36秒 → 提案 34560サンプル = 0.72秒
- 32kHz: 現在 12800サンプル = 0.40秒 → 提案 25600サンプル = 0.80秒

歌声のフレーズ構造を学習するには、より長いコンテキストが有効。日本語歌唱の1小節（テンポ120BPMで2秒）を完全にカバーはできないが、音素遷移パターンをより多くキャプチャできる。

#### 変更内容

**`configs/v2/48k.json` L12**
```json
"segment_size": 34560
```

**`configs/v2/32k.json` L12**
```json
"segment_size": 25600
```

#### VRAM使用量への影響
- `segment_size` 倍増によりVRAM使用量が増加
- RTX 4070 Ti Super 16GB: `batch_size=4` で対応可能な見込み（現在のbs=4から変更不要の見込み）
- もしOOMが発生した場合は `batch_size=2` に削減で対応
- 学習速度: 1エポックの所要時間は概ね倍増する見込み

#### 注意点
- `segment_size` は学習時のクロップ長であり、推論時には使われない → 既存モデル互換性に影響なし
- `train.py` L157: `hps.train.segment_size // hps.data.hop_length` でモデル初期化に使われるが、これはモデルの内部パラメータには影響しない（フレーム数の計算用）
- 前処理で生成されるセグメントの最小長（`preprocess.py` の `per` パラメータ）が `segment_size` より十分長いことを確認すること

---

### 1-6: 歌唱向け前処理パラメータ
**工数**: 1日 | **変更ファイル**: `infer/modules/train/preprocess.py`, `configs/config.py`, `infer/modules/vc/pipeline.py`

#### 背景
現在の前処理パラメータ（`PreProcess.__init__` L36-48）:

| パラメータ | 現在値 | コード上の位置 | 意味 |
|-----------|--------|---------------|------|
| `threshold` (Slicer) | -42 dB | L39 | 無音判定閾値 |
| `min_length` (Slicer) | 1500 ms | L40 | 最小セグメント長 |
| `min_interval` (Slicer) | 400 ms | L41 | 最小無音間隔 |
| `per` | 3.7 s | L36 (引数デフォルト), L47 (`self.per`) | セグメント分割長 |
| `overlap` | 0.3 s | L48 | セグメント間オーバーラップ |
| ハイパスフィルタ Wn | 48 Hz | L46 | カットオフ周波数 |

歌声データはスピーチに比べて、持続音が長く、息継ぎが短く、低周波成分が多い。そのため歌声に最適化したパラメータに変更する。

#### 変更内容

**`infer/modules/train/preprocess.py` -- `PreProcess.__init__`**

```python
# 変更前
self.slicer = Slicer(
    sr=sr,
    threshold=-42,
    min_length=1500,
    min_interval=400,
    hop_size=15,
    max_sil_kept=500,
)
# ...
self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr)
self.per = per
self.overlap = 0.3

# 変更後
self.slicer = Slicer(
    sr=sr,
    threshold=-38,
    min_length=2000,
    min_interval=400,
    hop_size=15,
    max_sil_kept=500,
)
# ...
self.bh, self.ah = signal.butter(N=5, Wn=40, btype="high", fs=self.sr)
self.per = per
self.overlap = 0.5
```

また、`infer-web.py` から `preprocess.py` を呼び出す際の `per` 引数のデフォルト値も変更が必要。`preprocess.py` L36 の `PreProcess.__init__` デフォルト引数:
```python
# 変更前
def __init__(self, sr, exp_dir, per=3.7):

# 変更後
def __init__(self, sr, exp_dir, per=5.0):
```

さらに `pipeline.py` L24 のグローバルハイパスフィルタも同期して変更:
```python
# 変更前
bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)

# 変更後
bh, ah = signal.butter(N=5, Wn=40, btype="high", fs=16000)
```

#### パラメータ変更の根拠

| パラメータ | 変更 | 根拠 |
|-----------|------|------|
| threshold | -42→-38 dB | 歌声のブレスや弱音部を無音と誤判定しないよう閾値を緩和 |
| per | 3.7→5.0 s | 歌声のフレーズ構造を保持。segment_size拡張と整合 |
| overlap | 0.3→0.5 s | セグメント境界でのフレーズ切断を軽減 |
| min_length | 1500→2000 ms | 短すぎるセグメントによる品質劣化を防止 |
| ハイパスWn | 48→40 Hz | 男声低音域（C2=65Hz付近）の基本周波数を保持。48Hzカットオフだと5次バターワースの傾斜により65Hz付近で-3dB以上の減衰が発生 |

**`configs/config.py` -- `preprocess_per` のデフォルト値も変更が必要**

`per` パラメータは実際には `configs/config.py` L57 の `self.preprocess_per = 3.7` が元であり、`infer-web.py` L210 で CLI引数として `preprocess.py` に渡される。`preprocess.py` のデフォルト引数だけ変更しても WebUI経由では効果がない。

```python
# configs/config.py L57
# 変更前
self.preprocess_per = 3.7

# 変更後
self.preprocess_per = 5.0
```

> **注意**: `config.py` L154 では GPU VRAM <= 4GB の場合に `self.preprocess_per = 3.0` に上書きされる。低VRAMユーザ向けの安全弁であり、この上書きは変更しない。

#### 注意点
- これらの変更は**新規前処理にのみ影響**し、既に前処理済みのデータには影響しない
- `pipeline.py` のグローバルハイパスフィルタ（L24）は推論時にも使われるため、既存モデルでの推論にも影響する。ただし40→48Hzの変更は可聴域外のDCオフセット除去目的なので、音質への実質的影響はない
- `per` パラメータはCLI引数 `sys.argv[6]` 経由でも渡される（L15）。上述の通り `configs/config.py` の `preprocess_per` が実際のソースであるため、両方の箇所を変更すること

---

### 1-8: Multi-Resolution STFT損失追加
**工数**: 1日 | **変更ファイル**: `infer/lib/train/losses.py`, `infer/modules/train/train.py`, `pyproject.toml`

#### 背景
現在の学習損失構成（`train.py` L464-468）:
```python
loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel    # メルスペクトログラムL1
loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl  # KLダイバージェンス
loss_fm = feature_loss(fmap_r, fmap_g)                        # 特徴量マッチング
loss_gen, losses_gen = generator_loss(y_d_hat_g)              # GAN生成器損失
loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
```

単一解像度のメルスペクトログラム損失のみでは、異なる時間-周波数解像度での再現性が保証されない。Multi-Resolution STFT (MR-STFT) 損失を追加することで、広帯域（高時間分解能）と狭帯域（高周波分解能）の両方の品質を同時に改善する。

#### 変更内容

**`pyproject.toml` -- 依存関係に追加**
```toml
"auraloss>=0.4.0",
```

**`infer/lib/train/losses.py` -- 末尾に追加**

> **注意**: 現在の `losses.py` は `torch` のみインポートしており `torch.nn` を使っていない。`nn.Module` を使うためインポート追加が必要。

```python
import torch.nn as nn
import auraloss.freq


class MultiResolutionSTFTLoss(nn.Module):
    """auralossベースのMulti-Resolution STFT損失"""

    def __init__(
        self,
        fft_sizes=(1024, 2048, 512),
        hop_sizes=(256, 512, 128),
        win_lengths=(1024, 2048, 512),
    ):
        # レビュー修正: hop_sizes, win_lengthsを実装に合わせて更新
        super().__init__()
        self.mrstft = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=list(fft_sizes),
            hop_sizes=list(hop_sizes),
            win_lengths=list(win_lengths),
        )

    def forward(self, y_hat, y):
        """
        Args:
            y_hat: (B, 1, T) - 生成音声
            y: (B, 1, T) - ターゲット音声
        Returns:
            loss: スカラー
        """
        return self.mrstft(y_hat, y)
```

**`infer/modules/train/train.py` -- インポート追加**
```python
from infer.lib.train.losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
    MultiResolutionSTFTLoss,  # 追加
)
```

**`infer/modules/train/train.py` -- `run()` 関数内、net_d初期化後（L171付近）に追加**
```python
mrstft_loss = MultiResolutionSTFTLoss()
if torch.cuda.is_available():
    mrstft_loss = mrstft_loss.cuda(rank)
```

**`infer/modules/train/train.py` -- `train_and_evaluate()` 関数の引数にmrstft_lossを追加して渡す**

`run()` 内の `train_and_evaluate()` 呼び出し（L250-276）を修正し、`mrstft_loss` を引数として渡す。

**`infer/modules/train/train.py` -- 損失計算部分（L460-468付近）**
```python
# 変更前
loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

# 変更後
loss_mrstft = mrstft_loss(y_hat, wave) * getattr(hps.train, 'c_mrstft', 5.0)
loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_mrstft
```

**`configs/v2/48k.json`, `configs/v2/32k.json` -- train セクションに追加**
```json
"c_mrstft": 5.0
```

#### 注意点
- `auraloss` はPyTorchのみに依存し、追加のバイナリ依存はない
- MR-STFT損失の重み `c_mrstft=5.0`（レビューで2.5から5.0に引き上げ）。学習効果の強化のため
- `getattr` フォールバックにより、旧JSONファイルでも動作する
- TensorBoardのログに `loss/g/mrstft` を追加して監視可能にすること

---

### 1-9: 歌声プリセット（WebUI）
**工数**: 1日 | **変更ファイル**: `infer-web.py`, 新規 `infer/lib/f0_presets.py`

#### 背景
日本語歌声変換では、ジャンルや用途に応じてF0パラメータを調整する必要がある。初心者ユーザにとってこれは困難なため、ワンクリックでパラメータを適用できるプリセット機能を追加する。

#### プリセット定義

**新規ファイル `infer/lib/f0_presets.py`**

```python
"""歌声変換プリセット定義"""

PRESETS = {
    "カスタム": {
        "description": "手動設定（プリセットなし）",
        "f0_method": None,
        "filter_radius": None,
        "f0_min": None,
        "f0_max": None,
    },
    "J-POP": {
        "description": "J-POP向け。ビブラート保存、標準音域",
        "f0_method": "rmvpe",
        "filter_radius": 1,
        "f0_min": 65,
        "f0_max": 1100,
    },
    "演歌": {
        "description": "演歌向け。こぶし保存、フィルタなし",
        "f0_method": "rmvpe",
        "filter_radius": 0,
        "f0_min": 65,
        "f0_max": 900,    # レビューで1100→900に修正
    },
    "アニソン": {
        "description": "アニソン向け。広音域、高速F0追従",
        "f0_method": "fcpe",
        "filter_radius": 1,
        "f0_min": 80,
        "f0_max": 1200,   # レビューで1400→1200に修正
    },
    "話し声": {
        "description": "話し声向け。従来互換パラメータ",
        "f0_method": "rmvpe",
        "filter_radius": 3,
        "f0_min": 50,
        "f0_max": 800,
    },
}


def get_preset_names():
    """プリセット名のリストを返す"""
    return list(PRESETS.keys())


def get_preset(name):
    """指定プリセットのパラメータ辞書を返す。存在しない場合は"カスタム"を返す"""
    return PRESETS.get(name, PRESETS["カスタム"])
```

#### WebUI側の変更

**`infer-web.py` -- 推論タブ内に「歌声プリセット」ドロップダウンを追加**

推論タブ（「単独ファイル推論」「バッチ推論」の両方）にプリセット選択UIを追加。プリセットを選択すると、`f0method`, `filter_radius` の値を自動的に書き換える。

```python
from infer.lib.f0_presets import get_preset_names, get_preset

# UI定義
preset_dropdown = gr.Dropdown(
    choices=get_preset_names(),
    value="カスタム",
    label="歌声プリセット",
    interactive=True,
)

# コールバック
def apply_preset(preset_name):
    p = get_preset(preset_name)
    if p["f0_method"] is None:
        # カスタム: 現在値を維持（何も変更しない）
        return gr.update(), gr.update()
    return gr.update(value=p["f0_method"]), gr.update(value=p["filter_radius"])

preset_dropdown.change(
    fn=apply_preset,
    inputs=[preset_dropdown],
    outputs=[f0method0, filter_radius0],
)
```

#### 注意点
- プリセット選択は推論パラメータのデフォルト値を変更するだけで、ロジックには影響しない
- 「カスタム」を選択すると何も変更されず、ユーザの手動設定が維持される
- `f0_min` / `f0_max` はプリセットに定義しているが、現在のUI上で直接変更するコントロールがない。1-2のF0レンジ拡張と合わせて、将来的にUI側にもmin/maxスライダーを追加することを検討（ただしM1スコープ外でもよい）
- i18n対応: プリセット名は日本語で定義しているが、多言語対応が必要な場合はi18n関数経由にする

---

### 1-10: bfloat16移行
**工数**: 0.5日 | **変更ファイル**: `configs/v2/48k.json`, `configs/v2/32k.json`, `infer/modules/train/train.py`

#### 背景
現在の設定:
- `fp16_run: true`（`configs/v2/48k.json` L10, `configs/v2/32k.json` L10）
- `train.py` L245: `GradScaler(enabled=hps.train.fp16_run)`
- `train.py` L409: `autocast("cuda", enabled=hps.train.fp16_run)`

fp16（float16）は値の範囲が狭く（最大65504）、音声生成のような微妙な値を扱うタスクではオーバーフローやアンダーフローが発生しやすい。bfloat16は指数部がfloat32と同じ8ビットのため、値の範囲はfloat32と同等でありながら、メモリ使用量はfp16と同じ。

RTX 4070 Ti Super (Ada Lovelace) はbfloat16をネイティブサポートしている。

#### 変更内容

**`configs/v2/48k.json`, `configs/v2/32k.json`**
```json
"fp16_run": true,
"bf16_run": true
```

**`infer/modules/train/train.py` -- autocast部分の変更**

```python
# 変更前
with autocast("cuda", enabled=hps.train.fp16_run):

# 変更後
use_bf16 = getattr(hps.train, 'bf16_run', False) and torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
with autocast("cuda", enabled=hps.train.fp16_run, dtype=amp_dtype):
```

**GradScaler の調整**
bfloat16使用時は `GradScaler` が不要（bfloat16はダイナミックレンジがfp32と同等のため、loss scalingが不要）:
```python
# 変更前
scaler = GradScaler(enabled=hps.train.fp16_run)

# 変更後
use_bf16 = getattr(hps.train, 'bf16_run', False) and torch.cuda.is_bf16_supported()
scaler = GradScaler(enabled=hps.train.fp16_run and not use_bf16)
```

#### 注意点
- bfloat16は**Ampere以降のGPU**（RTX 30xx以降）でのみネイティブサポート。古いGPUでは `torch.cuda.is_bf16_supported()` が `False` を返すため、自動的にfp16にフォールバック
- 推論時には影響しない（推論は `model.eval()` + `torch.no_grad()` で動作し、autocastは使わない）
- `is_half` フラグは推論用の精度制御で、学習時の `fp16_run` / `bf16_run` とは独立

---

### 1-11: 評価実行 + ベースライン比較
**工数**: 1日 | **変更ファイル**: なし（評価スクリプトはM0で構築済み前提）

#### 手順

1. **ベースライン測定**: M1の変更を適用する前に、現行パラメータで学習・推論し、以下のメトリクスを記録
   - MCD (Mel Cepstral Distortion)
   - F0 RMSE
   - 推論レイテンシ（音声1秒あたりの処理時間）
   - （M0で評価パイプラインが構築済みの場合はそれを使用）

2. **M1適用後の測定**: 全サブタスク適用後に同一条件で学習・推論し、同じメトリクスを測定

3. **比較レポート作成**: Go/No-Go判定基準に照らして結果を評価

4. **テストデータ**: 日本語歌声のテストセット（学習データと異なる曲・異なるフレーズ）

#### Go/No-Go判定基準（再掲）

| 基準 | Go条件 | No-Go時の対応 |
|------|--------|--------------|
| MCD | 5%以上改善 | パラメータ再調整 |
| F0 RMSE | 10%以上改善 | FCPEパラメータ微調整 |
| レイテンシ | 200ms以下 | segment_size調整 |
| 既存モデル互換 | 完全維持 | 変更箇所をリバート |

---

### 変更対象ファイル一覧

| ファイル | サブタスク | 変更種別 |
|---------|-----------|---------|
| `infer/modules/vc/pipeline.py` | 1-1, 1-2, 1-6 | F0メソッド追加、レンジ変更、ハイパスフィルタ変更 |
| `infer/modules/train/extract/extract_f0_print.py` | 1-1, 1-2 | F0メソッド追加、レンジ変更 |
| `infer-web.py` | 1-1, 1-3, 1-9 | F0メソッドUI追加、デフォルト値変更、プリセットUI追加 |
| `tools/infer_cli.py` | 1-3 | デフォルト値変更 |
| `tools/infer_batch_rvc.py` | 1-3 | デフォルト値変更 |
| `tools/app.py` | 1-3 | デフォルト値変更 |
| `configs/v2/48k.json` | 1-4, 1-5, 1-8, 1-10 | p_dropout, weight_decay, segment_size, c_mrstft, bf16_run |
| `configs/v2/32k.json` | 1-4, 1-5, 1-8, 1-10 | 同上 |
| `infer/modules/train/train.py` | 1-4, 1-8, 1-10 | weight_decay, MR-STFT損失, bf16対応 |
| `infer/modules/train/preprocess.py` | 1-6 | Slicerパラメータ、ハイパスフィルタ、per, overlap |
| `configs/config.py` | 1-6 | `preprocess_per` デフォルト値変更 (3.7→5.0) |
| `infer/lib/train/losses.py` | 1-8 | MultiResolutionSTFTLoss クラス追加 |
| `infer/lib/f0_presets.py` | 1-9 | **新規作成** |
| `pyproject.toml` | 1-8 | auraloss依存追加 |

---

### 技術仕様

#### FCPE仕様
- パッケージ: `torchfcpe>=0.0.4`（既にpyproject.tomlに記載済み）
- モデル: バンドル済み（追加ダウンロード不要）
- 入力: `(batch, samples, 1)` float32テンソル
- 出力: `(batch, frames)` F0値（Hz）
- `decoder_mode`: `"local_argmax"`（推奨）、`"argmax"`（高精度・低速）
- `threshold`: voicing判定閾値。デフォルト`0.006`

#### Multi-Resolution STFT損失の解像度設定（レビュー修正後）
| 解像度 | FFT size | Hop size | Window length | 特性 |
|--------|----------|----------|---------------|------|
| 広帯域 | 512 | 128 | 512 | 高時間分解能（過渡応答） |
| 中帯域 | 1024 | 256 | 1024 | バランス型 |
| 狭帯域 | 2048 | 512 | 2048 | 高周波分解能（定常音） |

#### bfloat16 vs float16

| 特性 | float16 | bfloat16 |
|------|---------|----------|
| 仮数部 | 10ビット | 7ビット |
| 指数部 | 5ビット | 8ビット |
| 最大値 | 65504 | 3.39e+38 |
| 精度 | 高い | やや低い |
| ダイナミックレンジ | 狭い | float32と同等 |
| GradScaler必要 | 要 | 不要 |

---

## 3. エージェントチーム構成

### 推奨構成（3エージェント）

| 役割 | 担当サブタスク | スキル要件 |
|------|---------------|-----------|
| **推論パイプラインエージェント** | 1-1, 1-2, 1-3, 1-9 | PyTorch推論、Gradio UI、F0アルゴリズム理解 |
| **学習パイプラインエージェント** | 1-4, 1-5, 1-6, 1-8, 1-10 | PyTorch学習ループ、損失関数設計、AMP |
| **評価エージェント** | 1-11 | メトリクス計算、A/Bテスト、レポート作成 |

### 依存関係と実行順序

```
Week 1:
  推論エージェント: 1-1 → 1-2 → 1-3 → 1-9
  学習エージェント: 1-4 → 1-5 → 1-6 → 1-8 → 1-10
  （並列実行可能）

Week 2:
  評価エージェント: 1-11（推論/学習エージェントの成果物に依存）
```

### レビュー体制
- 各サブタスク完了時にセルフレビュー
- Week 1終了時に推論/学習エージェント間でクロスレビュー
- Week 2終了時に全体レビュー + Go/No-Go判定

---

## 4. 提供範囲・テスト項目

### スコープ

#### In Scope
- 上記サブタスク1-1〜1-11の実装
- 既存のF0メソッド（pm, harvest, crepe, rmvpe）の動作維持
- 既存のWebUI/CLI推論フローの動作維持
- 既存の学習フローの動作維持（新パラメータでの改善含む）

#### Out of Scope
- mel_fmin変更（M3-Bに延期）
- モデルアーキテクチャの変更（M2以降）
- 日本語特化HuBERTの導入（M2以降）
- データ拡張パイプライン（M2以降）
- 新規事前学習モデルの作成（M3以降）
- DWTベースのビブラート保存フィルタ（M3-Aタスク3-5で実施予定）

### ユニットテスト

#### 1-1: FCPE統合
- [ ] `pipeline.py` の `get_f0(f0_method="fcpe")` が正常にF0配列を返すこと
- [ ] 返されるF0配列の長さが `p_len` と概ね一致すること（注: rmvpe/crepeも厳密には一致しない場合がある。`get_f0` の後続処理でリサイズされるため、大きな乖離がないことを確認）
- [ ] F0値が `f0_min` 〜 `f0_max` の範囲内（0除く）であること
- [ ] `extract_f0_print.py` の `compute_f0(f0_method="fcpe")` が正常動作すること
- [ ] `extract_f0_print.py` への `import torch` 追加により既存メソッド（pm, harvest, rmvpe）の動作が影響されないこと

#### 1-2: F0レンジ拡張
- [ ] `f0_min=65`, `f0_max=1400` で各F0メソッドが動作すること
- [ ] 1400Hz付近のテスト音声でF0が正しく検出されること
- [ ] `f0_coarse` の値域が 1〜255 に収まること

#### 1-3: filter_radius変更
- [ ] `filter_radius=1` でハーベスト使用時にメディアンフィルタが適用されないこと
- [ ] `filter_radius=3` を明示指定した場合に従来と同じ動作をすること
- [ ] WebUI, CLI, バッチ推論すべてでデフォルト値が1であること

#### 1-4: Dropout + Weight Decay
- [ ] `p_dropout=0.1` でモデルが正常に初期化されること
- [ ] `model.train()` 時にDropoutが有効であること（出力が非決定的）
- [ ] `model.eval()` 時にDropoutが無効であること（出力が決定的）
- [ ] 既存の `p_dropout=0` の学習済みモデルが `eval()` で正常に推論できること
- [ ] AdamWに `weight_decay=0.01` が適用されていること

#### 1-5: segment_size拡張
- [ ] 48kHz: `segment_size=34560` で学習が正常に開始されること
- [ ] 32kHz: `segment_size=25600` で学習が正常に開始されること
- [ ] batch_size=4でOOMが発生しないこと（RTX 4070 Ti Super 16GB）
- [ ] OOM発生時にbatch_size=2に変更して正常動作すること

#### 1-6: 前処理パラメータ
- [ ] 歌声データでSlicerが適切にセグメント分割すること（持続音が途中で切れない）
- [ ] 40Hzハイパスフィルタが65Hz付近の音声を過度に減衰させないこと
- [ ] `per=5.0` でセグメント長が適切であること
- [ ] `overlap=0.5` でセグメント境界のアーティファクトが軽減されること
- [ ] `configs/config.py` の `preprocess_per=5.0` がWebUI経由の前処理に正しく反映されること

#### 1-8: MR-STFT損失
- [ ] `MultiResolutionSTFTLoss` が `(B, 1, T)` のテンソルに対して正常にスカラー損失を返すこと
- [ ] `auraloss` のインポートが成功すること
- [ ] 損失値がTensorBoardに記録されること
- [ ] `c_mrstft` パラメータが設定ファイルから正しく読み込まれること

#### 1-9: プリセット
- [ ] 全プリセットが `get_preset()` で正しく返されること
- [ ] 「カスタム」選択時にパラメータが変更されないこと
- [ ] WebUIでプリセット変更時にUI要素が正しく更新されること
- [ ] 存在しないプリセット名指定時に「カスタム」にフォールバックすること

#### 1-10: bfloat16
- [ ] bfloat16対応GPUで `use_bf16=True` になること
- [ ] bfloat16使用時にGradScalerが無効化されること
- [ ] 非対応GPUで自動的にfp16にフォールバックすること
- [ ] 学習が正常に収束すること

### E2Eテスト

- [ ] **推論E2E**: 歌声WAVファイル → WebUI推論（各F0メソッド: pm, harvest, crepe, rmvpe, fcpe） → 出力WAV → MCD計算
- [ ] **学習E2E**: 歌声データ → 前処理 → F0抽出 → 学習（10エポック） → 推論 → 品質確認
- [ ] **プリセットE2E**: プリセット選択 → パラメータ自動適用 → 推論 → 出力品質確認
- [ ] **既存モデル互換E2E**: M1変更適用後のコードで、M1適用前に学習したモデルを読み込み → 推論 → 出力が変わらないことを確認

### テストデータ

| データ | 用途 | 備考 |
|--------|------|------|
| 日本語歌声（女声・J-POP）1曲 | 学習用 | 声優歌唱データ |
| 日本語歌声（女声・J-POP）別フレーズ | テスト用 | 学習データと異なるフレーズ |
| 日本語歌声（男声）1フレーズ | 音域テスト | 低音域テスト |
| 合成正弦波 (65Hz, 440Hz, 1400Hz) | F0検出テスト | レンジ境界テスト |
| 既存話し声データ | 回帰テスト | 話し声品質が劣化しないこと |

---

## 5. 懸念事項とレビュー項目

### 実装上の懸念

#### 懸念1: segment_size倍増によるOOM
- **リスク**: 中（RTX 4070 Ti Super 16GBでbatch_size=4がギリギリの可能性）
- **対策**: 事前にbatch_size=4でのVRAM使用量をプロファイリング。OOM発生時はbatch_size=2で代替
- **検出方法**: 学習開始直後（最初の数ステップ）でOOMが発生する

#### 懸念2: MR-STFT損失の重みバランス
- **リスク**: 中（レビューでc_mrstft=2.5→5.0に引き上げ済み）
- **対策**: TensorBoardで各損失項のスケールを監視。loss_mrstftが他の損失項と桁違いの場合は重み調整
- **検出方法**: 学習初期のTensorBoardログで確認

#### 懸念3: FCPE の GPU メモリ使用
- **リスク**: 低（推論時のみのため）
- **対策**: FCPEモデルは遅延ロード（`if not hasattr(self, "model_fcpe")`）で必要時のみ初期化
- **検出方法**: 推論時のGPUメモリ監視

#### 懸念4: 前処理パラメータ変更の副作用
- **リスク**: 低（新規学習にのみ影響）
- **対策**: ベースラインとの前後比較。セグメント分割数・セグメント長の統計を記録
- **検出方法**: 前処理ログの確認

#### 懸念5: bfloat16の数値精度
- **リスク**: 低（仮数部が7ビットと少ないが、音声タスクでは十分）
- **対策**: fp16での学習結果と比較して品質劣化がないことを確認
- **検出方法**: 学習収束後のMCD比較

#### 懸念6: pipeline.pyのグローバルハイパスフィルタ変更（48→40Hz）の推論影響
- **リスク**: 非常に低（可聴域下限のDCオフセット除去目的）
- **対策**: 変更前後の推論結果を波形レベルで比較。40-48Hz帯域のエネルギー差分を確認
- **検出方法**: A/B聴取テスト

### レビューチェックリスト

- [ ] 各サブタスクの変更がそのサブタスクのファイルスコープ内に収まっているか
- [ ] 新規追加コードにdocstringと型アノテーションがあるか
- [ ] `getattr` フォールバックにより旧設定JSONでも動作するか
- [ ] 全F0メソッド（pm, harvest, crepe, rmvpe, fcpe）で推論が正常動作するか
- [ ] `model.eval()` 時にDropoutが無効化されることを確認したか
- [ ] TensorBoardに追加メトリクス（loss_mrstft）が正しく記録されるか
- [ ] bfloat16非対応GPU（Ampere以前）で自動フォールバックするか
- [ ] プリセットの「カスタム」が既存動作に一切影響しないか
- [ ] `infer-web.py` のi18n対応（必要に応じて翻訳キーを追加）
- [ ] pyproject.toml に `auraloss>=0.4.0` が追記されているか
- [ ] ruffフォーマットに準拠しているか（`uv run ruff check .`）
- [ ] 既存モデル（`assets/pretrained_v2/`）での推論が変更前と同一結果を返すか

---

## 6. 一から作り直すとしたら

M0+M1フェーズ全体を一から設計し直すとしたら、以下のアプローチを取る。

### 構成の理想形

#### 1. 設定管理の統一
現在のコードベースでは設定が `configs/v2/*.json`、`sys.argv`、ハードコーディングの3箇所に分散している。理想的には:
- 全設定を `dataclass` ベースの型安全な設定オブジェクトに統一
- 学習・推論・前処理の設定をYAMLで一元管理
- プリセットも同じYAMLフォーマットで定義し、オーバーライドする形に

```python
@dataclass
class F0Config:
    method: str = "rmvpe"
    min_hz: float = 65.0
    max_hz: float = 1400.0
    filter_radius: int = 1

@dataclass
class TrainConfig:
    segment_size: int = 34560
    batch_size: int = 4
    p_dropout: float = 0.1
    weight_decay: float = 0.01
    precision: str = "bf16"  # "fp16", "bf16", "fp32"
    loss_weights: dict = field(default_factory=lambda: {
        "mel": 45.0, "kl": 1.0, "mrstft": 2.5
    })
```

#### 2. F0抽出の抽象化
現在は `get_f0` メソッド内に全メソッドの分岐が長大な if-elif チェーンで書かれている。理想的には:

```python
class F0Extractor(ABC):
    @abstractmethod
    def extract(self, audio: np.ndarray, sr: int) -> np.ndarray: ...

class RMVPEExtractor(F0Extractor): ...
class FCPEExtractor(F0Extractor): ...
class HarvestExtractor(F0Extractor): ...

# レジストリパターンで登録
F0_REGISTRY = {"rmvpe": RMVPEExtractor, "fcpe": FCPEExtractor, ...}
```

#### 3. 損失関数のプラグイン化
```python
class LossRegistry:
    def __init__(self):
        self.losses = {}

    def register(self, name, loss_fn, weight):
        self.losses[name] = (loss_fn, weight)

    def compute(self, **kwargs):
        total = 0
        details = {}
        for name, (fn, w) in self.losses.items():
            l = fn(**kwargs) * w
            total += l
            details[name] = l
        return total, details
```

#### 4. 評価パイプラインの組み込み
M0の評価パイプラインをライブラリとしてプロジェクトに組み込み、学習ループのバリデーションステップで自動的にMCD/F0 RMSEを計算・TensorBoardに記録する。

#### 5. 前処理の歌声/話声モード切替
前処理時に `--mode singing` / `--mode speech` フラグを受け取り、内部パラメータセットを自動切替。現在の「全ユーザ統一パラメータ」方式では、歌声ユーザと話し声ユーザのどちらかが犠牲になる。

### なぜ理想形にしないのか
- 既存コードベースの大規模リファクタリングはM1のスコープ外
- 既存モデル互換性を維持しながら段階的に改善するのがM1の方針
- アルゴリズム変更は基本的に受け付けていないというプロジェクトのコントリビューションルールに配慮
- 上記の理想形はM3以降で段階的に導入することを推奨

---

## 7. 後続タスクへの連絡事項

### M2に引き継ぐべき情報

#### 1. M1の評価結果
- ベースラインと改善後のMCD/F0 RMSE/レイテンシの数値
- 各サブタスクの個別効果（可能であれば段階的に適用して測定）
- Go/No-Go判定の結果と根拠

#### 2. FCPEの所見
- FCPE vs rmvpe のF0精度比較結果
- FCPE特有のアーティファクト（もしあれば）
- `threshold` / `decoder_mode` の最適値

#### 3. segment_size拡張の影響
- VRAM使用量の実測値（batch_sizeとの組み合わせ）
- 学習速度への影響（1エポックの所要時間）
- 品質への影響（MCD改善度）

#### 4. MR-STFT損失の知見
- `c_mrstft` の最終値と調整過程
- 損失バランスの推移（TensorBoardスクリーンショット）
- 他の損失項（c_mel, c_kl）との相互作用

#### 5. bfloat16の結果
- fp16との品質差（あれば）
- 学習安定性の違い
- GradScaler不要化による学習速度への影響

#### 6. mel_fmin変更（M3-Bへの申し送り）
- M1ではmel_fmin変更を見送った理由：事前学習済みモデルがmel_fmin=0.0で学習されており、変更すると既存モデルとの完全な互換性が失われる
- M3-Bで新規事前学習モデルを作成する際にmel_fmin=40.0を適用すること
- mel_fmin変更時の影響範囲: `configs/v2/*.json`、`mel_processing.py`、事前学習モデル全体
- **注意**: `milestones.md` のMVPセクションに `mel_fmin: 0.0 → 40.0` が「互換性維持」の設定変更として記載されているが、これは誤りである。mel_fmin変更は既存事前学習モデルと非互換のため、milestones.md側の修正が必要

#### 7. 前処理パラメータの最適値
- 歌声データでの前処理結果（セグメント数、セグメント長分布）
- threshold=-38dBでの誤分割・過剰結合の発生状況
- per=5.0sのセグメント長が学習品質に与えた影響

#### 8. 技術的負債
- `pipeline.py` の `get_f0` メソッドが5分岐のif-elifチェーンになっている。M2でF0抽出のクラス化を推奨
- 設定値のハードコーディング箇所が複数残っている（pipeline.py L24のハイパスフィルタ等）
- `preprocess.py` のCLI引数渡し（sys.argv）がフラジャイル。M2以降でargparseまたは設定ファイル化を推奨

#### 9. プリセット機能の拡張性
- M1では4プリセット（J-POP, 演歌, アニソン, 話し声）を定義
- M2以降でユーザカスタムプリセットの保存/読み込み機能を検討
- f0_min/f0_maxのUI化（スライダー追加）を検討

#### 10. milestones.md の要修正箇所
- **MVPセクション**: `mel_fmin: 0.0 → 40.0` は互換性を破壊するため、MVPリスト（互換性維持の設定変更）から削除すべき
- **M3-Aタスク3-1**: MRSTFT損失はM1（タスク1-8）に前倒し済みだが、M3-Aのタスクリストに残っている。M3-Aから削除するか「M1で実施済み」と注記すべき
