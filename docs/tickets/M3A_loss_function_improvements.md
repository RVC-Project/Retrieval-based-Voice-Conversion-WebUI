# M3-A: 損失関数改善

## メタ情報
- **マイルストーン**: M3
- **フェーズ**: Phase 3-A（Week 6）
- **工数見積もり**: 4人日
- **GPU要件**: RTX 4070 Ti Super 16GB（実装・推論テスト）、RTX 4090 24GB（学習検証）
- **前提タスク**: M2完了（SSL置換+日本語歌声事前学習済み）
- **ステータス**: 未着手
- **関連マイルストーン**: [milestones.md](../milestones.md) > M3 Phase 3-Aセクション
- **関連提案書**: [proposals_summary.md](../proposals_summary.md) > 提案3: 損失関数・学習手法改善
- **追加パッケージ**: `PyWavelets>=1.4.0`（pyproject.toml に追記）。`auraloss>=0.4.0` はM1（タスク1-8）で追加済みの前提

> **注意（MRSTFT損失との関係）**: milestones.md において、Multi-Resolution STFT損失（タスク3-1に相当）はM1のタスク1-8に前倒し済みである（レビュー反映: 「MRSTFT損失をM3から前倒し（auraloss追加+数行で実装可能、互換性維持）」）。M1チケット（`M1_immediate_improvements.md` タスク1-8）に`MultiResolutionSTFTLoss`クラスの実装仕様が記載されている。M1が完了している前提のため、本チケットのタスク3-1はM1実装の検証・係数チューニングに限定する。`auraloss>=0.4.0` はM1で追加済みの想定。

---

## 1. タスク目的とゴール

### 目的

M2で構築した日本語SSL+歌声事前学習モデルの上に、損失関数と学習手法の改善を重ね、音質指標をさらに向上させる。現行の学習ループ（`train.py` L464-468）は4つの損失（mel L1, KL, feature matching, GAN generator）で構成されているが、以下の課題がある。

1. **KL損失が学習初期にポステリア崩壊を誘発**: `c_kl=1.0` が固定値のため、学習初期にKLペナルティが潜在表現の多様性を過度に抑制し、生成音声の表現力が制限される
2. **チェックポイント間の品質ばらつき**: EMAなしで毎エポック保存されるため、チェックポイントごとに品質が不安定になり、ユーザーがどのチェックポイントを使うか判断に迷う
3. **学習率スケジューリングの単調性**: 現行の`ExponentialLR(gamma=0.999875)`は単調減衰であり、局所最適からの脱出能力が低い。特に少量データ（10分）でのファインチューニングでは、最適解を探索しきれないリスクがある
4. **ビブラート・しゃくりの周波数域情報損失**: mel L1損失は時間-周波数の平均誤差しか捉えず、ビブラートの微細な周波数変動（5-7Hz）やしゃくりの急激なピッチ遷移を明示的に保存する機構がない

### なぜこのタイミングか

- M2完了後のモデルは日本語歌声に最適化されたSSL特徴量と事前学習済みG/Dを持っており、損失関数改善の効果が最大化される時点である
- Phase 3-Aの全改善は既存モデルの重み構造に影響せず（推論グラフ不変）、学習プロセスのみの変更であるため、M2成果物との互換性を完全に維持できる
- 後続のPhase 3-B（ボコーダ改善）は事前学習再実行を伴うため、3-Aで学習手法を安定化させておくことが3-Bの効率的な開発に寄与する

### 成功条件

| 指標 | M2完了時点（推定） | M3-A目標 | 測定方法 |
|------|-------------------|----------|----------|
| MCD | 6.0-7.0 dB | 5.5-6.5 dB | `tools/eval/metrics/mcd.py` |
| F0 RMSE | 12-20 cents | 10-18 cents | `tools/eval/metrics/f0_accuracy.py` |
| Whisper CER | 8-15% | 7-12% | `tools/eval/metrics/whisper_cer.py` |
| 主観MOS | 3.5-4.0 | 3.8-4.2 | 開発者3名の5段階評価 |
| チェックポイント品質分散 | 未測定 | 標準偏差50%削減（EMA効果） | 5チェックポイントのMCD標準偏差 |

---

## 2. 実装する内容の詳細

### サブタスク一覧

| # | タスク | 工数 | 互換性 | 変更ファイル |
|---|--------|------|--------|-------------|
| 3-1 | Multi-Resolution STFT損失（検証・チューニング） | 0.5日 | 維持 | `train.py`, `configs/v2/*.json` |
| 3-2 | KLサイクリカルアニーリング | 0.5日 | 維持 | `train.py` |
| 3-3 | EMA（alpha=0.999） | 1日 | 維持 | `train.py`, `infer/lib/train/utils.py` |
| 3-4 | CosineAnnealingWarmRestarts + Warmup | 1日 | 維持 | `train.py`, `configs/v2/*.json` |
| 3-5 | DWTビブラート保存損失 | 1日 | 維持 | `infer/lib/train/losses.py`, `train.py`, `pyproject.toml` |

---

### 3-1: Multi-Resolution STFT損失（検証・チューニング）
**工数**: 0.5日 | **互換性**: 維持

#### 前提

M1のタスク1-8で `MultiResolutionSTFTLoss` が `infer/lib/train/losses.py` に実装済みであり、`train.py` に統合済みの想定。本タスクではM2事前学習モデルとの組み合わせで最適な係数を検証する。

#### 現状（M1実装後の想定コード）

```python
# train.py L468付近
loss_mrstft = mrstft_loss(y_hat, wave) * getattr(hps.train, 'c_mrstft', 2.5)
loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_mrstft
```

#### 作業内容

1. **係数グリッドサーチ**: `c_mrstft` を `[1.0, 2.5, 5.0, 10.0]` の4パターンで短時間学習（各50エポック）し、MCD/CERを比較
2. **MRSTFT解像度パラメータの見直し**: M1のデフォルト `fft_sizes=(1024, 2048, 512)` が48kHz歌声に最適か検証。必要に応じて `(2048, 4096, 1024)` など高解像度設定を試行
3. **TensorBoardログの確認**: `loss/g/mrstft` が正常にログされているか確認。損失のスケールが他の損失項と大きくずれていないか確認

#### 変更ファイル

| ファイル | 変更内容 |
|----------|----------|
| `configs/v2/48k.json` | `c_mrstft` の値を検証結果に基づき更新 |
| `configs/v2/32k.json` | 同上 |
| `train.py` | MRSTFT解像度パラメータの調整（必要な場合のみ） |

#### 注意点
- M1で `auraloss>=0.4.0` が `pyproject.toml` に追加済みでない場合は追加すること
- M1未実装の場合は、M1チケットのタスク1-8の仕様に従い実装を行うこと（工数0.5日→1日に増加）

---

### 3-2: KLサイクリカルアニーリング
**工数**: 0.5日 | **互換性**: 維持

#### 背景

現行の学習ループでは `c_kl=1.0` が学習全体を通じて固定である（`configs/v2/48k.json` L16）。VITSベースのモデルではKL損失が学習初期に潜在空間の分布を正規分布に強く近づけるため、エンコーダの表現力が制約される「ポステリア崩壊」が発生しうる。サイクリカルアニーリングにより、KL重みを0から徐々に1.0まで上昇させ、1サイクル完了後にリセットすることで潜在空間のより多様な表現を獲得する。

#### 現行コード（`train.py` L465）

```python
loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
```

#### 変更内容

**`train.py` -- `run()` 関数内、スケジューラ初期化後（L244付近）に追加**

```python
# KLサイクリカルアニーリング設定
kl_anneal_epochs = getattr(hps.train, 'kl_anneal_epochs', 100)  # 1サイクルのエポック数
kl_anneal_strategy = getattr(hps.train, 'kl_anneal_strategy', 'cyclical')  # 'cyclical' or 'monotonic'
```

**`train.py` -- 学習ループ内（`for epoch in range(epoch_str, ...)` L248 の直後）にKL重み計算を追加**

`kl_weight` は `epoch` から計算でき、`epoch` は既に `train_and_evaluate()` の引数であるため、**`train_and_evaluate()` の引数変更は不要**。`train_and_evaluate()` 内の損失計算前に以下を追加する。

```python
# KL重みの計算（train_and_evaluate() 内、損失計算ループの前に配置）
kl_anneal_epochs = getattr(hps.train, 'kl_anneal_epochs', 100)
kl_anneal_strategy = getattr(hps.train, 'kl_anneal_strategy', 'cyclical')
if kl_anneal_strategy == 'cyclical':
    cycle_position = (epoch % kl_anneal_epochs) / kl_anneal_epochs
    kl_weight = min(1.0, cycle_position * 2.0)  # 前半で0→1に上昇、後半は1.0維持
elif kl_anneal_strategy == 'monotonic':
    kl_weight = min(1.0, epoch / kl_anneal_epochs)
else:
    kl_weight = 1.0
```

**`train.py` -- 損失計算部分（L465）を変更**

```python
# 変更前
loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

# 変更後
loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl * kl_weight
```

**`train.py` -- TensorBoardログ（L491-503付近の `scalar_dict` 更新後）に追加**

```python
scalar_dict.update({"loss/kl_weight": kl_weight})
```

#### 設定パラメータ（`configs/v2/48k.json`, `32k.json` の `train` セクション）

```json
"kl_anneal_epochs": 100,
"kl_anneal_strategy": "cyclical"
```

`getattr` フォールバックにより、パラメータ未設定時はデフォルト値が使用されるため既存config互換性は維持される。

#### 理論的根拠

- Fu et al. (2019) "Cyclical Annealing Schedule" で提案された手法
- VAE/VITS系モデルの標準的なテクニック
- ファインチューニング時（10-50エポック程度）では `kl_anneal_epochs=20` に短縮を推奨

---

### 3-3: EMA（Exponential Moving Average, alpha=0.999）
**工数**: 1日 | **互換性**: 維持

#### 背景

EMAは学習中のパラメータの指数移動平均を保持し、推論時にEMAパラメータを使用することで、チェックポイント間の品質ばらつきを大幅に低減する。現行の学習ループではチェックポイント保存時に直近の勾配更新が反映されたパラメータをそのまま使用しており、特に小バッチ学習（batch_size=4）ではチェックポイント品質が不安定になりやすい。

#### 実装方針

PyTorch標準の `torch.optim.swa_utils.AveragedModel` の使用を検討するが、RVCの `DDP` ラッピングとの互換性を考慮し、手動実装を採用する。

#### 変更内容

**`infer/lib/train/utils.py` -- EMAヘルパークラスを追加**

```python
import copy

class EMA:
    """Exponential Moving Average for model parameters."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register(model)

    def _register(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        """EMAパラメータをモデルに適用（推論/保存用）"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    def restore(self, model):
        """元のパラメータに復元（学習続行用）"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name].clone()
        self.backup = {}

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict
```

**`train.py` -- `run()` 関数内、DDP初期化後に追加**

```python
# EMA初期化
ema_decay = getattr(hps.train, 'ema_decay', 0.0)  # 0.0=無効、0.999=有効
if ema_decay > 0:
    ema_g = EMA(net_g.module if hasattr(net_g, 'module') else net_g, decay=ema_decay)
else:
    ema_g = None
```

**`train.py` -- `train_and_evaluate()` のオプティマイザstep直後に追加**

```python
# EMA更新（Generator のみ）
if ema_g is not None:
    ema_g.update(net_g.module if hasattr(net_g, 'module') else net_g)
```

**`train.py` -- チェックポイント保存時（L523-574: 定期保存、L578-589: 最終保存の両方）を変更**

チェックポイント保存は2箇所ある。`save_every_epoch`（L523-574）と最終エポック保存（L578-589）の両方でEMAを適用する必要がある。

```python
# --- 定期保存（L523-574）および最終保存（L578-589）の両方に適用 ---

# EMAパラメータで保存
if ema_g is not None:
    model_for_save = net_g.module if hasattr(net_g, 'module') else net_g
    ema_g.apply_shadow(model_for_save)

# --- 既存の保存コード ---
if hasattr(net_g, "module"):
    ckpt = net_g.module.state_dict()
else:
    ckpt = net_g.state_dict()
# ...savee(ckpt, ...) or utils.save_checkpoint(net_g, ...)...

# EMA解除（学習続行のため。最終保存時は不要だが統一的に呼ぶ）
if ema_g is not None:
    ema_g.restore(model_for_save)
```

> **重要**: `utils.save_checkpoint()` は内部で `model.module.state_dict()` を呼び出すため、`ema_g.apply_shadow()` を先に呼べば `utils.save_checkpoint()` も自動的にEMAパラメータを保存する。`savee()` によるweight保存（L554-574）も同様。

**`train.py` -- EMA状態のチェックポイント保存/復元**

学習再開時にEMA状態を復元するため、チェックポイントにEMA shadowを含める。

```python
# 保存時（定期保存ブロック内、utils.save_checkpoint の直後）
if ema_g is not None:
    ema_path = os.path.join(hps.model_dir, "ema_g.pt")
    torch.save({"shadow": ema_g.state_dict(), "decay": ema_g.decay}, ema_path)

# 復元時（run() 関数内、EMA初期化直後）
if ema_g is not None:
    ema_path = os.path.join(hps.model_dir, "ema_g.pt")
    if os.path.exists(ema_path):
        ema_state = torch.load(ema_path, map_location="cpu", weights_only=False)
        ema_g.load_state_dict(ema_state["shadow"])
        logger.info("Loaded EMA state from %s" % ema_path)
```

#### 設定パラメータ（`configs/v2/48k.json`, `32k.json` の `train` セクション）

```json
"ema_decay": 0.999
```

`getattr` フォールバックにより、未設定時は `ema_decay=0.0`（EMA無効）となるため既存config互換。

#### 注意点

- EMAはGeneratorのみに適用する。Discriminatorにはパラメータの安定化が不要であり、むしろGAN学習のダイナミクスを阻害しうる
- `ema_decay=0.999` は一般的な推奨値。短いファインチューニング（10-50エポック）では `0.995` に下げることを検討
- EMAパラメータのshadowコピーにより、VRAM使用量が Generator パラメータ分（約55MB）増加する。RTX 4070 Ti Super 16GBでは問題ない
- `DDP` 環境では `net_g.module` でアンラップしてからEMA操作を行うこと。Intel XPU環境（L188-189）ではDDPラッピングがスキップされるため、`hasattr(net_g, 'module')` による分岐が必須
- EMA state_dict の保存/復元ロジックは、チェックポイントからの学習再開時にも対応させること（上記の `ema_g.pt` 方式）

---

### 3-4: CosineAnnealingWarmRestarts + Warmup
**工数**: 1日 | **互換性**: 維持

#### 背景

現行のスケジューラは `ExponentialLR(gamma=0.999875)` である（`train.py` L242-243）。なお `warmup_epochs` は既に `configs/v2/48k.json` L14 と `32k.json` L14 に `0` として定義されているが、現行コードでは使用されていない。

```python
scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
```

この単調減衰スケジューラには以下の問題がある。
- 局所最適からの脱出能力が低い（一度下がった学習率は戻らない）
- 長期学習（数百エポック）で学習率が極端に小さくなり学習が事実上停止する
- 少量データFTでは最初のウォームアップなしに高い学習率で開始され、初期の不安定な勾配更新が品質を損なう

CosineAnnealingWarmRestartsに置換し、線形ウォームアップを組み合わせることで、周期的な学習率回復と安定した学習開始を実現する。

#### 変更内容

**`train.py` -- 新規ヘルパー関数を追加**

```python
def build_scheduler(optimizer, hps_train, last_epoch):
    """学習率スケジューラを構築する。

    hps_train.scheduler_type に基づきスケジューラを選択:
      - 'exponential' (デフォルト、後方互換): ExponentialLR
      - 'cosine_warm_restarts': CosineAnnealingWarmRestarts + 線形Warmup
    """
    scheduler_type = getattr(hps_train, 'scheduler_type', 'exponential')

    if scheduler_type == 'cosine_warm_restarts':
        T_0 = getattr(hps_train, 'scheduler_T_0', 50)       # 最初のサイクル長（エポック）
        T_mult = getattr(hps_train, 'scheduler_T_mult', 2)   # サイクル長の倍率
        eta_min = getattr(hps_train, 'scheduler_eta_min', 1e-6)  # 最小学習率
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min, last_epoch=last_epoch
        )
    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=hps_train.lr_decay, last_epoch=last_epoch
        )

    return scheduler
```

**`train.py` -- Warmupラッパーの実装**

```python
class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """線形ウォームアップ付きスケジューララッパー

    注意: super().__init__() は内部で get_lr() を呼び出すため、
    self.warmup_epochs と self.base_scheduler を先に設定する必要がある。
    """

    def __init__(self, optimizer, warmup_epochs, base_scheduler, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        # 重要: super().__init__() が get_lr() を呼ぶため、上記2行の後に呼ぶこと
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 線形ウォームアップ: 0 → base_lr
            alpha = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            return self.base_scheduler.get_last_lr()

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epochs:
            self.base_scheduler.step(epoch)
        super().step(epoch)
```

> **チェックポイント再開時の注意**: `epoch_str > warmup_epochs` の場合、ウォームアップは自動的にスキップされる（`get_lr()` で `self.last_epoch >= self.warmup_epochs` の分岐に入る）。ただし `last_epoch=epoch_str - 2` で初期化されるため、再開時のウォームアップ再実行は発生しない。

**`train.py` -- L242-243を変更**

```python
# 変更前
scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

# 変更後
warmup_epochs = getattr(hps.train, 'warmup_epochs', 0)
base_scheduler_g = build_scheduler(optim_g, hps.train, last_epoch=epoch_str - 2)
base_scheduler_d = build_scheduler(optim_d, hps.train, last_epoch=epoch_str - 2)

if warmup_epochs > 0:
    scheduler_g = WarmupScheduler(optim_g, warmup_epochs, base_scheduler_g, last_epoch=epoch_str - 2)
    scheduler_d = WarmupScheduler(optim_d, warmup_epochs, base_scheduler_d, last_epoch=epoch_str - 2)
else:
    scheduler_g = base_scheduler_g
    scheduler_d = base_scheduler_d
```

#### 設定パラメータ（`configs/v2/48k.json`, `32k.json` の `train` セクション）

```json
"scheduler_type": "cosine_warm_restarts",
"scheduler_T_0": 50,
"scheduler_T_mult": 2,
"scheduler_eta_min": 1e-6,
"warmup_epochs": 5
```

> **注意**: `warmup_epochs` は既存の `configs/v2/48k.json` L14 および `32k.json` L14 に `0` として定義済みだが、現行コードでは参照されていない。本タスクで `WarmupScheduler` を実装することで初めて機能する。既存configの `warmup_epochs: 0` はウォームアップ無効を意味するため後方互換。

`getattr` フォールバックにより、`scheduler_type` 等の新パラメータが未設定時は `scheduler_type='exponential'`（既存ExponentialLR）にフォールバックするため完全互換。

#### TensorBoardログ

`learning_rate` は既にログされているが（`train.py` L478, L492）、ウォームアップ/コサインの動作を確認するためにログ出力が正しいか確認すること。

#### 推奨パラメータ

| 用途 | scheduler_type | T_0 | T_mult | warmup_epochs |
|------|---------------|-----|--------|---------------|
| ファインチューニング（短期） | cosine_warm_restarts | 20 | 1 | 3 |
| ファインチューニング（標準） | cosine_warm_restarts | 50 | 2 | 5 |
| 事前学習 | cosine_warm_restarts | 100 | 2 | 10 |
| 既存互換 | exponential | - | - | 0 |

---

### 3-5: DWTビブラート保存損失
**工数**: 1日 | **互換性**: 維持

#### 背景

ビブラートは歌声の重要な表現要素で、典型的には5-7Hzの周波数でF0が変動する。しゃくりやこぶしなどの装飾的技法も含め、これらはmel L1損失では明示的に保存されない。Discrete Wavelet Transform (DWT) を用いて信号を異なる周波数帯域に分解し、各帯域ごとの誤差を計算することで、ビブラート帯域の再現性を強化する。

#### 変更内容

**`pyproject.toml` -- 依存関係に追加**

```toml
"PyWavelets>=1.4.0",
```

**`infer/lib/train/losses.py` -- 末尾に追加**

> **注意**: 現在の `losses.py` は `torch` のみインポートしている（L1）。`nn.Module` と `F.conv1d` 等を使うため、`torch.nn` と `torch.nn.functional` のインポートを追加する必要がある。M1のタスク1-8で `import torch.nn as nn` が追加済みの想定だが、`torch.nn.functional as F` はM1では追加されないため本タスクで追加する。

```python
import torch.nn as nn
import torch.nn.functional as F  # DWT損失で使用（F.pad, F.conv1d, F.l1_loss）

try:
    import pywt

    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False


class DWTVibrateLoss(nn.Module):
    """DWT（離散ウェーブレット変換）ベースのビブラート保存損失

    信号をDWTで多解像度分解し、各周波数帯域でL1誤差を計算する。
    ビブラート帯域（5-7Hz相当のサブバンド）に追加重みを付与することで、
    ビブラート・しゃくり・こぶしなどの歌唱表現の保存を促進する。

    Args:
        wavelet: ウェーブレット名（デフォルト: 'db4'）
        level: 分解レベル（デフォルト: 4）
        vibrato_weight: ビブラート帯域の追加重み（デフォルト: 2.0）
    """

    def __init__(self, wavelet='db4', level=4, vibrato_weight=2.0):
        super().__init__()
        if not HAS_PYWT:
            raise ImportError("PyWavelets is required for DWTVibrateLoss: pip install PyWavelets")
        self.wavelet = wavelet
        self.level = level
        self.vibrato_weight = vibrato_weight

        # DWTフィルタ係数をバッファとして登録
        wavelet_obj = pywt.Wavelet(wavelet)
        dec_lo = torch.tensor(wavelet_obj.dec_lo, dtype=torch.float32)
        dec_hi = torch.tensor(wavelet_obj.dec_hi, dtype=torch.float32)
        self.register_buffer('dec_lo', dec_lo)
        self.register_buffer('dec_hi', dec_hi)

    def _dwt_1level(self, x):
        """1レベルのDWT分解（畳み込みベース、GPU対応）

        Args:
            x: (B, 1, T)
        Returns:
            approx: (B, 1, T//2) - 近似係数
            detail: (B, 1, T//2) - 詳細係数
        """
        filt_len = self.dec_lo.shape[0]
        # パディング
        pad_size = filt_len - 1
        x_padded = F.pad(x, (pad_size, pad_size), mode='reflect')
        # フィルタを畳み込みカーネルに整形
        lo = self.dec_lo.flip(0).unsqueeze(0).unsqueeze(0)  # (1, 1, K)
        hi = self.dec_hi.flip(0).unsqueeze(0).unsqueeze(0)  # (1, 1, K)
        # 畳み込み + ダウンサンプリング (stride=2)
        approx = F.conv1d(x_padded, lo.to(x.device), stride=2)
        detail = F.conv1d(x_padded, hi.to(x.device), stride=2)
        return approx, detail

    def _multi_level_dwt(self, x):
        """マルチレベルDWT分解

        Args:
            x: (B, 1, T)
        Returns:
            coeffs: [detail_1, detail_2, ..., detail_L, approx_L]
                detail_1 = 最初に分離される高周波成分（最高周波数帯）
                detail_L = 最後に分離される低周波の詳細成分
                approx_L = 最低周波数帯の近似係数
        """
        details = []
        current = x
        for _ in range(self.level):
            current, detail = self._dwt_1level(current)
            details.append(detail)
        details.append(current)  # 最終近似係数
        return details

    def forward(self, y_hat, y):
        """
        Args:
            y_hat: (B, 1, T) - 生成音声
            y: (B, 1, T) - ターゲット音声
        Returns:
            loss: スカラー
        """
        # 長さを揃える
        min_len = min(y_hat.shape[-1], y.shape[-1])
        y_hat = y_hat[..., :min_len]
        y = y[..., :min_len]

        coeffs_hat = self._multi_level_dwt(y_hat)
        coeffs_ref = self._multi_level_dwt(y)

        loss = torch.tensor(0.0, device=y.device)
        n_coeffs = len(coeffs_hat)
        for i, (c_hat, c_ref) in enumerate(zip(coeffs_hat, coeffs_ref)):
            band_loss = F.l1_loss(c_hat, c_ref)
            # 周波数帯域の対応（level=4, 48kHz の場合）:
            #   details[0] = level 1: 12000-24000Hz（最高周波数帯）
            #   details[1] = level 2: 6000-12000Hz
            #   details[2] = level 3: 3000-6000Hz
            #   details[3] = level 4: 1500-3000Hz
            #   details[4] = approx:  0-1500Hz（DC〜低周波近似）
            #
            # ビブラート（5-7Hz）は波形ドメインでは振幅変調として現れ、
            # 基本周波数帯（1500-6000Hz程度、歌声のフォルマント帯域）の
            # エンベロープに反映される。低〜中周波のディテール係数
            # （level 3-4 = details[2], details[3]）とapprox（details[4]）に
            # 重みを置くことで、ビブラートの振幅変動パターンとフォルマント
            # 構造の保存を促進する。
            if i >= n_coeffs - 3:  # 最低3帯域（level 3, level 4, approx）にビブラート重み
                band_loss = band_loss * self.vibrato_weight
            loss = loss + band_loss

        return loss / n_coeffs
```

**`train.py` -- インポートに追加**

```python
from infer.lib.train.losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
    MultiResolutionSTFTLoss,  # M1で追加済み
    DWTVibrateLoss,           # M3-Aで追加
)
```

**`train.py` -- `run()` 関数内、損失関数初期化部分に追加**

```python
# DWTビブラート保存損失
c_dwt = getattr(hps.train, 'c_dwt', 0.0)  # 0.0=無効
if c_dwt > 0:
    dwt_loss_fn = DWTVibrateLoss(
        wavelet=getattr(hps.train, 'dwt_wavelet', 'db4'),
        level=getattr(hps.train, 'dwt_level', 4),
        vibrato_weight=getattr(hps.train, 'dwt_vibrato_weight', 2.0),
    )
    if torch.cuda.is_available():
        dwt_loss_fn = dwt_loss_fn.cuda(rank)
else:
    dwt_loss_fn = None
```

**`train.py` -- 損失計算部分に追加（L468付近）**

```python
# DWTビブラート保存損失
if dwt_loss_fn is not None:
    loss_dwt = dwt_loss_fn(y_hat, wave) * c_dwt
else:
    loss_dwt = 0.0

# 総損失
loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_mrstft + loss_dwt
```

**`train.py` -- TensorBoardログに追加**

```python
if dwt_loss_fn is not None:
    scalar_dict.update({"loss/g/dwt": loss_dwt})
```

#### 設定パラメータ（`configs/v2/48k.json`, `32k.json` の `train` セクション）

```json
"c_dwt": 1.0,
"dwt_wavelet": "db4",
"dwt_level": 4,
"dwt_vibrato_weight": 2.0
```

`getattr` フォールバックにより、未設定時は `c_dwt=0.0`（DWT損失無効）となるため完全互換。

#### 注意点

- `PyWavelets` はCPUライブラリだが、DWTフィルタ係数を `register_buffer` でGPUに載せ、`F.conv1d` ベースで演算するためGPU上で動作する。`F` は `torch.nn.functional` であり、本タスクで `losses.py` にインポートを追加する
- DWT分解は計算量が軽い（mel spectrogram計算と比較して無視できる程度）
- `try/except` による `pywt` のオプショナルインポートにより、PyWavelets未インストール環境でも `c_dwt=0.0` であれば問題なく動作する
- `vibrato_weight=2.0` は初期値。演歌等ビブラートが重要なジャンルでは `3.0-5.0` に引き上げることを検討

---

## 3. エージェントチーム構成

### 推奨チーム配置

| ロール | 担当タスク | スキル要件 |
|--------|-----------|-----------|
| **リード開発者** | 3-3 (EMA), 3-4 (スケジューラ) | PyTorch学習ループ、分散学習(DDP) |
| **ML研究者** | 3-2 (KLアニーリング), 3-5 (DWT損失) | 損失関数設計、信号処理 |
| **QA/検証担当** | 3-1 (MRSTFTチューニング), 全体E2Eテスト | 評価パイプライン、A/Bテスト |

### 並列化

```
Day 1: [3-1 MRSTFTチューニング] + [3-2 KLアニーリング]  ← 並列可
Day 2: [3-3 EMA 前半]          + [3-5 DWT損失 前半]      ← 並列可
Day 3: [3-3 EMA 後半]          + [3-5 DWT損失 後半]      ← 並列可
Day 4: [3-4 スケジューラ]       + [統合テスト・チューニング]
```

- タスク3-1と3-2は独立しており初日に並列実施可能
- タスク3-3（EMA）と3-5（DWT損失）は変更箇所が異なるため並列可能（EMAは`utils.py`+`train.py`保存部分、DWTは`losses.py`+`train.py`損失計算部分）
- タスク3-4はスケジューラ置換であり他の損失関数改善との統合テストを兼ねて最終日に実施

---

## 4. 提供範囲・テスト項目

### ユニットテスト

各テストは `tests/train/` ディレクトリに配置する。

#### 3-1: MRSTFT検証テスト

```python
# tests/train/test_mrstft_tuning.py
def test_mrstft_loss_coefficients():
    """c_mrstft の異なる値で損失が適切にスケールされることを確認"""

def test_mrstft_resolution_configs():
    """カスタムFFTサイズで損失が計算できることを確認"""
```

#### 3-2: KLアニーリングテスト

```python
# tests/train/test_kl_annealing.py
def test_cyclical_kl_weight():
    """サイクリカルアニーリングのKL重みが期待通りに変化することを確認"""
    # epoch=0 → kl_weight=0.0
    # epoch=50 (半サイクル) → kl_weight=1.0
    # epoch=100 (1サイクル完了) → kl_weight=0.0 (リセット)
    # epoch=150 → kl_weight=1.0

def test_monotonic_kl_weight():
    """モノトニックアニーリングのKL重みが0→1に単調増加することを確認"""

def test_default_kl_weight():
    """未設定時にkl_weight=1.0（固定）となることを確認"""
```

#### 3-3: EMAテスト

```python
# tests/train/test_ema.py
def test_ema_update():
    """EMA更新でshadowパラメータが指数移動平均に従うことを確認"""

def test_ema_apply_and_restore():
    """apply_shadow/restoreでパラメータが正しく切り替わることを確認"""

def test_ema_disabled_by_default():
    """ema_decay=0.0 の場合にEMAが無効であることを確認"""

def test_ema_state_dict():
    """EMAの保存/復元が正しく動作することを確認"""

def test_ema_checkpoint_resume():
    """ema_g.pt の保存/復元で shadow パラメータが一致することを確認"""

def test_ema_applied_to_all_save_paths():
    """定期保存(save_every_epoch)と最終保存(total_epoch)の両方でEMAが適用されることを確認"""
```

#### 3-4: スケジューラテスト

```python
# tests/train/test_scheduler.py
def test_build_scheduler_exponential():
    """デフォルト設定でExponentialLRが返されることを確認"""

def test_build_scheduler_cosine():
    """scheduler_type='cosine_warm_restarts'でCosineAnnealingWarmRestartsが返されることを確認"""

def test_warmup_scheduler():
    """ウォームアップ期間中に学習率が0→base_lrへ線形増加することを確認"""

def test_warmup_then_cosine():
    """ウォームアップ後にコサイン減衰が正しく動作することを確認"""

def test_warmup_skip_on_resume():
    """last_epoch > warmup_epochs での初期化時にウォームアップが再実行されないことを確認"""

def test_build_scheduler_fallback():
    """scheduler_type未設定時にExponentialLRがデフォルトで返されることを確認"""
```

#### 3-5: DWT損失テスト

```python
# tests/train/test_dwt_loss.py
def test_dwt_loss_shape():
    """任意のバッチサイズ・長さで損失がスカラーとして返ることを確認"""

def test_dwt_loss_zero_for_identical():
    """同一入力で損失が0に近いことを確認"""

def test_dwt_vibrato_weight():
    """vibrato_weightの変更で損失値が適切にスケールされることを確認"""

def test_dwt_loss_optional_import():
    """PyWavelets未インストール時にc_dwt=0.0で問題なく動作することを確認"""

def test_dwt_gpu_computation():
    """GPU上でDWT分解が正しく動作することを確認（CUDAが利用可能な場合のみ）"""

def test_dwt_vibrato_band_weighting():
    """低〜中周波帯域（level 3-4, approx）に正しく追加重みが適用されることを確認"""
```

### E2Eテスト（統合テスト）

```python
# tests/train/test_m3a_integration.py
def test_full_training_loop_with_m3a():
    """全M3-A改善を有効化した状態で5エポックの学習が完走することを確認
    - KLアニーリング有効
    - EMA有効
    - CosineAnnealingWarmRestarts有効
    - DWT損失有効
    - 学習損失が発散せず減少傾向にあること
    - EMAチェックポイントが正しく保存されること
    """

def test_backward_compatibility():
    """M3-A新設定パラメータ未設定の既存configで学習が従来通り動作することを確認
    - ExponentialLRがデフォルトで使用される
    - EMAが無効
    - KL重みが1.0固定
    - DWT損失が無効
    """

def test_checkpoint_resume_with_m3a():
    """M3-A有効化チェックポイントからの学習再開が正常に動作することを確認
    - EMA状態が復元される
    - スケジューラ状態が復元される
    """
```

### 品質評価テスト

| テスト | 手順 | 合格基準 |
|--------|------|----------|
| ベースライン比較 | M2チェックポイントでFT（M3-A無効 vs 有効、各50エポック） | MCD 5%以上改善 |
| EMA効果測定 | EMA有効/無効の5チェックポイントでMCD標準偏差を比較 | 標準偏差50%削減 |
| ビブラート品質 | 演歌・J-POPサンプルのビブラートRate/Extent測定 | Rate 5-7Hz維持、Extent差20%以内 |
| 収束速度 | 各スケジューラで損失収束までのエポック数を比較 | Cosine: 20%高速化 |

---

## 5. 懸念事項とレビュー項目

### 懸念事項

| # | 懸念 | 影響度 | 発生確率 | 対策 |
|---|------|--------|---------|------|
| 1 | **損失項の相互干渉**: 6損失（mel + KL + FM + GAN + MRSTFT + DWT）のバランスが崩れ、特定の損失が支配的になる | 高 | 中 | 全損失のスケールをTensorBoardで監視。各損失が同じオーダー（0.1-10程度）に収まるよう係数を調整 |
| 2 | **KLアニーリングの過剰な自由度**: KL重み=0の期間にポステリアエンコーダが暴走し、潜在空間が無意味な分布になる | 中 | 低 | サイクル前半で0→1に上昇するため完全にKL=0の期間は短い。問題があれば下限 `kl_weight_min=0.1` を設定 |
| 3 | **EMAとDDPの非互換**: `net_g.module` のアンラップが特定のPyTorchバージョンで正しく動作しない | 中 | 低 | `hasattr(net_g, 'module')` による分岐を徹底。PyTorch 2.10で動作検証 |
| 4 | **WarmupSchedulerとチェックポイント再開の整合性**: 学習再開時にウォームアップが再実行されてしまう | 高 | 低 | `last_epoch=epoch_str - 2` で初期化されるため、`epoch_str > warmup_epochs` の場合は `get_lr()` が自動的にウォームアップをスキップする |
| 4b | **`scheduler.step()` の呼び出し位置**: 現行コードでは `scheduler_g.step()` と `scheduler_d.step()` が `train_and_evaluate()` の外（L277-278）で呼ばれる。`WarmupScheduler.step()` 内で `base_scheduler.step()` も呼ぶ設計のため、二重ステップにならないよう注意 | 中 | 中 | `WarmupScheduler` は `_LRScheduler` を継承しており、L277の `scheduler_g.step()` が `WarmupScheduler.step()` を呼び、その中で `base_scheduler.step()` も呼ばれる。これが正しい動作であることを確認すること |
| 5 | **DWT損失のGPUメモリ**: 大きなsegment_sizeでDWT分解のメモリ使用量が増加 | 低 | 低 | segment_size=34560（M1変更後）でも追加メモリは数MB程度。問題ない見込み |
| 6 | **全改善を同時導入する際の切り分け困難**: 品質改善の要因が特定できない | 中 | 高 | 各改善をconfig項目でon/off可能に設計済み。A/Bテスト時は1項目ずつ有効化して効果測定 |

### コードレビューチェックリスト

- [ ] `getattr` フォールバックにより全新設パラメータが未設定時にデフォルト値を返すこと
- [ ] 既存の `configs/v2/48k.json`、`configs/v2/32k.json` を変更せずに学習が動作すること
- [ ] `DDP` 環境（multi-GPU）での `net_g.module` アンラップが正しいこと。Intel XPU環境（DDP未使用）でも動作すること
- [ ] EMAの `apply_shadow` / `restore` が保存コード全箇所で正しく呼ばれること:
  - L523-553: `save_every_epoch` による定期保存（`utils.save_checkpoint()`）
  - L554-574: `save_every_weights` による推論用weight保存（`savee()`）
  - L578-589: 最終エポック保存（`savee()`）
- [ ] EMA状態（`ema_g.pt`）がチェックポイントと共に保存・復元されること
- [ ] TensorBoardログに新しい損失項（`loss/g/dwt`, `loss/kl_weight`）が追加されていること
- [ ] `PyWavelets` 未インストール環境で `c_dwt=0.0` の場合にエラーが発生しないこと
- [ ] `losses.py` に `import torch.nn.functional as F` が追加されていること（DWT損失が使用）
- [ ] チェックポイントからの学習再開時にEMA状態、スケジューラ状態が正しく復元されること
- [ ] `WarmupScheduler.__init__` で `self.warmup_epochs` と `self.base_scheduler` が `super().__init__()` より前に設定されること
- [ ] `scheduler.step()` が二重に呼ばれないこと（L277-278 の外部呼び出しと `WarmupScheduler.step()` 内部）
- [ ] 学習時間の増加が20%以内に収まること（全損失有効時）
- [ ] 推論パス（`model.infer()`）に一切の変更がないこと（互換性維持）

---

## 6. 一から作り直すとしたら（M3フェーズ全体の理想設計）

仮にM3の損失関数・ボコーダ改善を白紙から設計し直すとしたら、以下のアプローチを取る。

### 統合損失フレームワーク

現行の `losses.py` は4つの独立した関数として実装されており、損失の追加・削除・重み付けの変更に毎回 `train.py` の修正が必要。理想的には、宣言的な損失構成システムを構築する。

```python
# 理想的な設計: configで損失を宣言的に定義
{
  "losses": {
    "mel_l1":     {"weight": 45.0, "enabled": true},
    "kl":         {"weight": 1.0, "enabled": true, "annealing": "cyclical", "cycle_epochs": 100},
    "feature_matching": {"weight": 2.0, "enabled": true},
    "gan_generator":    {"weight": 1.0, "enabled": true},
    "mrstft":     {"weight": 2.5, "enabled": true, "fft_sizes": [1024, 2048, 512]},
    "dwt_vibrato": {"weight": 1.0, "enabled": true, "wavelet": "db4", "level": 4},
    "f0_consistency": {"weight": 0.5, "enabled": false},
    "formant_preservation": {"weight": 0.3, "enabled": false}
  }
}
```

```python
# 理想的な LossManager クラス
class LossManager(nn.Module):
    """Config駆動型の損失関数マネージャ"""
    def __init__(self, loss_config):
        super().__init__()
        self.losses = nn.ModuleDict()
        for name, cfg in loss_config.items():
            if cfg.get('enabled', True):
                self.losses[name] = self._build_loss(name, cfg)

    def forward(self, **kwargs):
        total_loss = 0
        loss_dict = {}
        for name, loss_fn in self.losses.items():
            loss_val = loss_fn(**kwargs) * self.weights[name]
            total_loss += loss_val
            loss_dict[name] = loss_val
        return total_loss, loss_dict
```

この設計により以下が実現する。
- `train.py` を変更せずにconfigのみで損失の追加・削除・重み変更が可能
- A/Bテスト時にconfigの差し替えだけで比較実験が実行できる
- 各損失のon/offが明示的で、実験の再現性が向上

### 学習ループの宣言的構成

EMA、スケジューラ、勾配蓄積なども同様にconfig駆動にする。

```json
{
  "optimizer": {
    "type": "adamw",
    "lr": 1e-4,
    "betas": [0.8, 0.99],
    "weight_decay": 0.01
  },
  "scheduler": {
    "type": "cosine_warm_restarts",
    "T_0": 50,
    "T_mult": 2,
    "warmup_epochs": 5
  },
  "ema": {
    "enabled": true,
    "decay": 0.999
  },
  "gradient_accumulation_steps": 4
}
```

### なぜ今回この設計にしないか

- 既存コードの構造（`train.py` の手続き的な学習ループ）との乖離が大きく、変更範囲が大幅に広がる
- 他のマイルストーン（M3-B, M4）との整合性検証が必要
- 「コントリビューションルールとしてアルゴリズム変更は基本的に受け付けていない」という制約があり、大規模リファクタリングは慎重に進める必要がある
- 4人日の工数では理想設計の実装+テストは不可能

現実的なアプローチとして、本チケットでは `getattr` フォールバックによる既存互換維持を最優先とし、将来のリファクタリング（M4以降）で統合的な設計に移行する可能性を残す。

---

## 7. 後続タスクへの連絡事項

### M3-B（ボコーダ改善, Week 7-8）への引き継ぎ

1. **EMAの事前学習対応**: M3-Bでは事前学習の再実行（SnakeBeta + mel_fmin=40Hz）が必要。事前学習スクリプトでもEMAを有効化できるよう、EMAクラスの実装が `utils.py` に配置されていることを確認
2. **CosineAnnealingWarmRestartsの事前学習設定**: 事前学習用configファイル（`configs/v2/48k_pretrain.json`）にもスケジューラ設定を追加。事前学習では `T_0=100, T_mult=2, warmup_epochs=10` を推奨
3. **DWT損失とSnakeBetaの相互作用**: SnakeBeta導入後にDWT損失の係数 `c_dwt` の再チューニングが必要になる可能性がある。SnakeBeta版の事前学習モデルではビブラート再現性が変化するため
4. **config分離方針の維持**: M3-Aで追加した新設パラメータは全て `getattr` フォールバック付きであり、M3-Bの新規configでのみ有効化すれば既存互換を損なわない

### M4（高度な最適化）への情報

1. **損失フレームワークのリファクタリング**: セクション6で記述した `LossManager` の統合設計はM4で検討する価値がある。M3-Aの実装パターン（`getattr`フォールバック + config項目追加）が増え続けると保守性が低下するため
2. **F0量子化ビン拡大（256→512）との関連**: M4候補タスク4-1とDWT損失の組み合わせにより、ビブラート精度がさらに向上する見込み。DWT損失の係数を再調整すること
3. **勾配蓄積の検討**: M3-Aスコープからは除外したが、`proposals_summary.md` 提案3のPhase 3で記載されている勾配蓄積（4ステップ、実効バッチサイズ16）は、M4でバッチサイズに制約がある場合（VRAM不足時）に検討する

### config追加パラメータまとめ

M3-A完了時に `configs/v2/48k.json` および `32k.json` の `train` セクションに追加されるパラメータ一覧。

| パラメータ | デフォルト（未設定時） | 推奨値 | 用途 |
|-----------|---------------------|--------|------|
| `c_mrstft` | 2.5 | 検証結果による | MRSTFT損失の重み（M1で追加済み） |
| `kl_anneal_epochs` | 100 | 100 (事前学習), 20 (FT) | KLアニーリング1サイクルのエポック数 |
| `kl_anneal_strategy` | `'cyclical'` | `'cyclical'` | KLアニーリング戦略 |
| `ema_decay` | 0.0 (無効) | 0.999 | EMA減衰率 |
| `scheduler_type` | `'exponential'` | `'cosine_warm_restarts'` | スケジューラ種別 |
| `scheduler_T_0` | 50 | 50 (FT), 100 (事前学習) | コサインサイクル初期長 |
| `scheduler_T_mult` | 2 | 2 | サイクル長の倍率 |
| `scheduler_eta_min` | 1e-6 | 1e-6 | 最小学習率 |
| `warmup_epochs` | 0 | 5 (FT), 10 (事前学習) | ウォームアップエポック数（既存configに定義済み、本タスクで機能実装） |
| `c_dwt` | 0.0 (無効) | 1.0 | DWT損失の重み |
| `dwt_wavelet` | `'db4'` | `'db4'` | ウェーブレット種類 |
| `dwt_level` | 4 | 4 | DWT分解レベル |
| `dwt_vibrato_weight` | 2.0 | 2.0 | ビブラート帯域の追加重み |

### 変更ファイル一覧

| ファイル | タスク | 変更内容 |
|----------|--------|----------|
| `infer/modules/train/train.py` | 3-1~3-5 | 損失計算、EMA、スケジューラ、KLアニーリング |
| `infer/lib/train/losses.py` | 3-5 | `DWTVibrateLoss` クラス追加 |
| `infer/lib/train/utils.py` | 3-3 | `EMA` クラス追加 |
| `configs/v2/48k.json` | 3-1~3-5 | 新設パラメータ追加 |
| `configs/v2/32k.json` | 3-1~3-5 | 同上 |
| `pyproject.toml` | 3-5 | `PyWavelets>=1.4.0` 追加 |
| `tests/train/test_kl_annealing.py` | 3-2 | 新規テスト |
| `tests/train/test_ema.py` | 3-3 | 新規テスト |
| `tests/train/test_scheduler.py` | 3-4 | 新規テスト |
| `tests/train/test_dwt_loss.py` | 3-5 | 新規テスト |
| `tests/train/test_m3a_integration.py` | 全体 | 統合テスト |
