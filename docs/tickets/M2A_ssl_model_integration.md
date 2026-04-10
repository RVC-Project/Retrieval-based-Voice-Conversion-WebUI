# M2-A: SSLモデル統合

## メタ情報
- **マイルストーン**: M2
- **フェーズ**: Phase 2-A（Week 3）
- **工数見積もり**: 10.5人日（タスク2-1〜2-6c合計。milestones.md記載の8人日はタスク2-6b/2-6c追加前の値）
- **GPU要件**: RTX 4070 Ti Super 16GB（実装・推論テスト）、RTX 4090 24GB（学習パイプライン検証）
- **前提タスク**: M0完了（評価スクリプト `tools/eval/run_eval.py` が必要）、M1完了（ContentVec ベースラインCER値が必要）
- **ステータス**: 未着手
- **関連マイルストーン**: [milestones.md](../milestones.md) > M2 Phase 2-Aセクション

---

## 1. タスク目的とゴール

### 目的

現在のRVC WebUIは、SSLモデルとして `assets/hubert/hubert_base.pt`（ContentVec / HuBERT Base）を fairseq 経由でロードし、768次元の特徴量（layer 12出力）をハードコードで利用している。この設計を抽象化し、日本語に特化したSSLモデル（第1候補: `imprt/kushinada-hubert-base`）に差し替え可能にすることで、日本語歌声変換の品質（特にCER = 歌詞認識精度）を向上させる。

### 達成基準

1. **SSLモデルの抽象化**: fairseq 依存を解消し、HuggingFace transformers ベースのローダーに移行。新旧モデルを設定1つで切り替え可能にする
2. **768次元ハードコードの解消**: `models.py` の `TextEncoder` に渡す `in_channels=768` をパラメータ化し、任意の次元のSSLモデルに対応
3. **output_layer の設定化**: 現在 `output_layer=12` がパイプライン全箇所にハードコードされているのを、SSLモデル設定から読み取るようにする
4. **CER改善の実証**: kushinada-hubert-base を使った推論で、ContentVec比 CER 5pt以上の改善を確認
5. **既存モデルとの互換性維持**: 既存の `.pth` チェックポイント（ContentVec 768次元で学習済み）がそのまま推論できること

### 非ゴール

- 事前学習モデルの再学習（M2-Bで実施）
- MERT等の音楽特化SSLモデルの統合（本フェーズでは検証のみ）
- fairseq の完全削除（DMLパッチ等の代替策が確立するまで `pyproject.toml` からは外さない）

---

## 2. 実装する内容の詳細

### 2-1. SSLモデルローダー抽象化（2日）

**目的**: fairseq 固有の `checkpoint_utils.load_model_ensemble_and_task()` を抽象レイヤで包み、transformers / fairseq / カスタムモデルを統一インターフェースでロード可能にする。

**変更対象ファイル**: `infer/modules/vc/utils.py`（既存の `load_hubert()` を拡張）

**現状のコード** (`infer/modules/vc/utils.py`):
```python
from fairseq import checkpoint_utils

def load_hubert(config):
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["assets/hubert/hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()
```

**設計方針**:

```python
# infer/modules/vc/utils.py に追加

from dataclasses import dataclass
from typing import Optional

@dataclass
class SSLModelConfig:
    """SSLモデルの設定"""
    name: str                    # モデル識別名（UI表示用）
    backend: str                 # "fairseq" | "transformers"
    model_path: str              # ローカルパスまたは HuggingFace モデルID
    output_layer: int            # 特徴量抽出レイヤ（例: 12）
    ssl_dim: int                 # 出力特徴量次元（例: 768）
    normalize_input: bool        # 入力正規化の要否

# 組み込みプリセット
SSL_PRESETS = {
    "contentvec": SSLModelConfig(
        name="ContentVec (HuBERT Base)",
        backend="fairseq",
        model_path="assets/hubert/hubert_base.pt",
        output_layer=12,
        ssl_dim=768,
        normalize_input=False,
    ),
    "kushinada": SSLModelConfig(
        name="Kushinada HuBERT (日本語)",
        backend="transformers",
        model_path="imprt/kushinada-hubert-base",
        output_layer=12,
        ssl_dim=768,
        normalize_input=False,
    ),
    "rinna": SSLModelConfig(
        name="Rinna Japanese HuBERT",
        backend="transformers",
        model_path="rinna/japanese-hubert-base",
        output_layer=12,
        ssl_dim=768,
        normalize_input=False,
    ),
}

def load_ssl_model(config, ssl_config: Optional[SSLModelConfig] = None) -> torch.nn.Module:
    """
    SSLモデルをロードする統一インターフェース。
    ssl_config が None の場合は従来の ContentVec をロード（後方互換）。
    返されるモデルは extract_features(source, padding_mask, output_layer) を持つ。
    """
    if ssl_config is None:
        ssl_config = SSL_PRESETS["contentvec"]

    if ssl_config.backend == "fairseq":
        return _load_fairseq_model(config, ssl_config)
    elif ssl_config.backend == "transformers":
        return _load_transformers_model(config, ssl_config)
    else:
        raise ValueError(f"Unknown SSL backend: {ssl_config.backend}")
```

**互換性のポイント**:
- 既存の `load_hubert(config)` は `load_ssl_model(config)` のエイリアスとして残す（呼び出し元を段階的に移行可能にする）
- 返されるモデルオブジェクトは `extract_features(**inputs)` インターフェースを保持する（`pipeline.py` の `model.extract_features(**inputs)` 呼び出しをそのまま使えるように）

**呼び出し元の特定（全箇所）**:

| ファイル | 行 | 現在のロード方法 |
|---------|-----|-----------------|
| `infer/modules/vc/utils.py` | L22-33 | `load_hubert()` - fairseq 直接 |
| `infer/modules/vc/modules.py` | L155 | `load_hubert(self.config)` を呼び出し |
| `infer/lib/rtrvc.py` | L95-98 | `fairseq.checkpoint_utils` 直接呼び出し |
| `tools/rvc_for_realtime.py` | L91-94 | `fairseq.checkpoint_utils` 直接呼び出し |
| `infer/modules/train/extract_feature_print.py` | L87-91 | `fairseq.checkpoint_utils` 直接呼び出し |
| `infer/lib/jit/get_hubert.py` | L249-254 | `load_model_ensemble_and_task()` 直接 |

**DMLパッチ適用箇所**（fairseq import + `GradMultiply.forward` モンキーパッチ）:

| ファイル | import行 | パッチ行 |
|---------|---------|---------|
| `infer-web.py` | L24 | L64 |
| `infer/lib/rtrvc.py` | L8 | L65 |
| `tools/rvc_for_realtime.py` | L8 | L64 |
| `infer/modules/train/extract_feature_print.py` | L21 | L43 |

> **注意**: `infer-web.py` 自体もfairseqをimportしDMLパッチを適用している。transformers移行後もDMLパッチのためにfairseq importを維持する必要がある（SSLモデルのロード自体は `utils.py` 経由に統一されるが、DMLパッチは各エントリーポイントで独立に適用される設計）。

---

### 2-2. HuggingFace transformers ラッパー（1日）

**目的**: `transformers.HubertModel` をラップして、既存の fairseq HuBERT と同じ `extract_features()` インターフェースを提供する。

**新規ファイル**: `infer/modules/vc/ssl_wrapper.py`

**設計方針**:

```python
# infer/modules/vc/ssl_wrapper.py

import torch
import torch.nn as nn
from transformers import HubertModel, AutoConfig

class TransformersHuBERTWrapper(nn.Module):
    """
    HuggingFace transformers の HubertModel を fairseq 互換の
    extract_features() インターフェースで包むラッパー。
    """
    def __init__(self, model_name_or_path: str, device: str = "cpu"):
        super().__init__()
        self.model = HubertModel.from_pretrained(
            model_name_or_path,
            output_hidden_states=True,
        )
        self.model.eval()

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: torch.Tensor,
        output_layer: int = 12,
    ):
        """
        fairseq HuBERT 互換のインターフェース。
        
        Args:
            source: (batch, time) の波形テンソル
            padding_mask: (batch, time) のBoolTensor
            output_layer: 何番目の Transformer layer の出力を使うか
            
        Returns:
            tuple: (features, padding_mask)
                features: (batch, time_frames, hidden_dim) のテンソル
        """
        # transformers の HubertModel は attention_mask（padding_mask の反転）を受け取る
        attention_mask = ~padding_mask if padding_mask is not None else None
        
        outputs = self.model(
            input_values=source,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # hidden_states[0] = CNN特徴量, [1]~[12] = Transformer layer 1~12
        # output_layer=12 なら hidden_states[12] を返す
        features = outputs.hidden_states[output_layer]
        return (features, padding_mask)
```

**検証項目**:
- ContentVec の fairseq 版と transformers 版で、同一音声に対する特徴量出力の差分が十分小さい（cosine similarity > 0.99）ことを確認
- kushinada-hubert-base の初回ダウンロード（HuggingFace Hub からの自動DL）が正常動作すること
- half precision（fp16）で推論が通ること

**依存追加**: `pyproject.toml` に `transformers>=4.40.0` を追加。ただし `fairseq` はまだ削除しない（DMLパッチが fairseq 依存のため）。

---

### 2-3. 推論パイプライン対応 - output_layer 設定化（1日）

**目的**: `output_layer=12` のハードコードを全箇所で設定値に置き換える。

**変更対象ファイル**:
- `infer/modules/vc/pipeline.py` (L194)
- `infer/lib/rtrvc.py` (L352)
- `tools/rvc_for_realtime.py` (L345)

**現状** (`infer/modules/vc/pipeline.py` L189-199):
```python
inputs = {
    "source": feats.to(self.device),
    "padding_mask": padding_mask,
    "output_layer": 12,  # ← ハードコード
}
t0 = ttime()
with torch.no_grad():
    logits = model.extract_features(**inputs)
    feats = logits[0]
```

**変更後**:
```python
inputs = {
    "source": feats.to(self.device),
    "padding_mask": padding_mask,
    "output_layer": self.ssl_output_layer,  # SSLModelConfig から設定
}
```

**`Pipeline.__init__` の変更**:
```python
class Pipeline(object):
    def __init__(self, tgt_sr, config, ssl_config=None):
        # ... 既存のコード ...
        # SSL設定
        if ssl_config is None:
            from infer.modules.vc.utils import SSL_PRESETS
            ssl_config = SSL_PRESETS["contentvec"]
        self.ssl_output_layer = ssl_config.output_layer
        self.ssl_dim = ssl_config.ssl_dim
```

**`rtrvc.py` の変更** (L334-355):
- `RVC.__init__` に `ssl_config` パラメータを追加
- `self.ssl_output_layer` をインスタンス変数として保持
- `infer()` メソッド内の `"output_layer": 12`（L352）を `self.ssl_output_layer` に置換

**`tools/rvc_for_realtime.py` の変更** (L334-346):
- `rtrvc.py` と同じパターン（`output_layer`: L345）。このファイルは `rtrvc.py` の `RVC` クラスのほぼ完全なコピーであり、独立した `RVC` クラスを持っている。`rtrvc.py` の修正と同じ変更を個別に適用する必要がある。

---

### 2-4. 学習パイプライン対応（1日）

**目的**: 特徴量抽出スクリプトでSSLモデルを切り替え可能にする。

**変更対象ファイル**: `infer/modules/train/extract_feature_print.py`

**現状** (L55-97):
```python
model_path = "assets/hubert/hubert_base.pt"  # ← ハードコード
# ...
models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
    [model_path],
    suffix="",
)
model = models[0]
# ...
inputs = {
    "source": ...,
    "padding_mask": ...,
    "output_layer": 12,  # ← ハードコード（L121）
}
```

**変更方針**:
- コマンドライン引数に `--ssl_model` を追加（デフォルト: `contentvec`）
- `load_ssl_model()` を使ってモデルをロード
- `output_layer` を `SSLModelConfig` から取得
- 出力ディレクトリ名を `3_feature768` から `3_feature{ssl_dim}` に動的化（既存パスとの互換性のため `ssl_dim=768` の場合は `3_feature768` を維持）

**呼び出し元** (`infer-web.py` L322-330):
```python
cmd = '"%s" infer/modules/train/extract_feature_print.py %s %s %s %s "%s/logs/%s" %s %s' % (
    config.python_cmd,
    config.device,
    leng,
    idx,
    n_g,
    now_dir,
    ...
)
```
→ `--ssl_model` 引数を追加で渡すよう修正。

**`infer-web.py` のファイルリスト生成部分** (L428, L470, L559, L599):
```python
feature_dir = "%s/3_feature768" % (exp_dir)  # ← ハードコード
fea_dim = 768  # ← ハードコード
index = faiss.index_factory(768, "IVF%s,Flat" % n_ivf)  # ← ハードコード
```
→ SSLモデル設定に応じて動的に変更。ただし今回のターゲットSSLモデル（kushinada, rinna）はすべて768次元のため、実際の値変更は発生しない。将来の拡張に備えた設計変更。

---

### 2-5. モデル定義の ssl_dim パラメータ化（2日）

**目的**: `SynthesizerTrnMs768NSFsid` / `SynthesizerTrnMs768NSFsid_nono` の `TextEncoder(768, ...)` ハードコードを解消する。

**変更対象ファイル**: `infer/lib/infer_pack/models.py`

**現状** (`models.py` L609-610, L766-767):
```python
# SynthesizerTrnMs768NSFsid.__init__ 内
self.enc_p = TextEncoder(
    768,  # ← ハードコード: SSLモデルの出力次元
    inter_channels,
    ...
)

# SynthesizerTrnMs768NSFsid_nono.__init__ 内
self.enc_p = TextEncoder(
    768,  # ← 同上
    inter_channels,
    ...
)
```

**変更方針**:

```python
class SynthesizerTrnMs768NSFsid(nn.Module):
    def __init__(
        self,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        spk_embed_dim,
        gin_channels,
        sr,
        **kwargs,
    ):
        super().__init__()
        # ssl_dim はkwargsから取得。未指定時は768（後方互換）
        ssl_dim = kwargs.pop("ssl_dim", 768)
        # ... 既存コード ...
        self.enc_p = TextEncoder(
            ssl_dim,  # ← パラメータ化
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            float(p_dropout),
        )
```

**互換性ロジック**（最重要）:

既存の `.pth` チェックポイントは `cpt["config"]` にモデル構築引数をリストとして保存している。`ssl_dim` はこのリストに含まれていないため、チェックポイントロード時に明示的にデフォルト値を補完する必要がある。

```python
# infer/modules/vc/modules.py の get_vc() 内（L74-83相当）
self.cpt = torch.load(person, map_location="cpu", weights_only=False)
# 既存チェックポイントの互換処理
ssl_dim = self.cpt.get("ssl_dim", 768)  # 旧チェックポイントは768と見なす

if self.if_f0:
    self.net_g = SynthesizerTrnMs768NSFsid(
        *self.cpt["config"],
        is_half=self.config.is_half,
        ssl_dim=ssl_dim,
    )
else:
    self.net_g = SynthesizerTrnMs768NSFsid_nono(
        *self.cpt["config"],
        ssl_dim=ssl_dim,
    )
```

> **注意**: `SynthesizerTrnMs768NSFsid` は `kwargs["is_half"]` を `GeneratorNSF` に渡す（`models.py` L629）ため `is_half` kwarg が必須だが、`SynthesizerTrnMs768NSFsid_nono` は `is_half` を使用しない。現行コード（`modules.py` L81-83）と同じパターンを維持すること。
>
> また、`modules.py` L55-61の「sid空文字列時のモデルキャッシュクリア」処理内にも `SynthesizerTrnMs768NSFsid` / `_nono` の呼び出しがある（L58-60）。こちらにも `ssl_dim=ssl_dim` を追加する必要がある。

**設定ファイルへの反映** (`configs/v2/48k.json`, `configs/v2/32k.json`):

現在の設定JSONには `ssl_dim` フィールドが存在しない。以下を `model` セクションに追加する:
```json
{
  "model": {
    "ssl_dim": 768,
    ...
  }
}
```

ただし、既存の `SynthesizerTrnMs768NSFsid` のコンストラクタは設定JSONの `model` セクションではなく `cpt["config"]` リストから引数を展開する設計になっているため、設定JSONへの `ssl_dim` 追加は新規事前学習時のリファレンス用途。推論時は `cpt.get("ssl_dim", 768)` で後方互換を取る。

**事前学習用config**: 新しいSSLモデルで事前学習を行う場合は `configs/v2/48k_pretrain.json` を別ファイルとして作成し、`ssl_dim` や `ssl_model` フィールドを追加する（既存configは変更しない）。

---

### 2-6. kushinada vs ContentVec 検証（0.5日）

**目的**: kushinada-hubert-base が ContentVec 比で日本語歌声変換品質を改善するか定量評価する。

**手順**:

1. M0で構築した評価パイプラインを使用
2. 評価データ: M0で定義したテストセット
3. 評価指標: CER（主指標）、MCD（副指標）、ViSQOL（知覚品質）
4. 比較対象:
   - ContentVec (`assets/hubert/hubert_base.pt`) + 既存学習済みモデル → ベースラインCER（M1で取得済み）
   - kushinada (`imprt/kushinada-hubert-base`) + 同一モデルで推論 → CER
5. 判定基準:
   - CER 5pt以上改善 → kushinada を採用、M3の事前学習に進む
   - CER 5pt未満改善 → rinna/japanese-hubert-base でも同様の検証を追加実施
   - いずれも不足 → Spin V2 または MERT を候補として検討（ただしM2スコープ外）

**注意**: この検証はSSLモデルの差し替えのみで行う（事前学習の再実行は行わない）。つまり、ContentVecで事前学習されたRVCモデルの特徴量抽出部分だけをkushinadaに差し替えて推論する形になる。特徴量空間の不一致によりCERが悪化する可能性があるが、SSLモデルの日本語表現力の差を見るためのラフスクリーニングとして位置づける。本格的な評価はM3の事前学習後に実施。

---

### 2-6b. WebUI SSLモデル選択UI（1日）

**目的**: ユーザーがWebUI上でSSLモデルを選択できるようにする。

**変更対象ファイル**: `infer-web.py`

**UI設計**:

推論タブ:
```
SSLモデル: [ContentVec (HuBERT Base) ▼]  # Dropdown
           - ContentVec (HuBERT Base)      # デフォルト
           - Kushinada HuBERT (日本語)
           - Rinna Japanese HuBERT
```

学習タブ:
```
SSLモデル: [ContentVec (HuBERT Base) ▼]  # 特徴量抽出時に使用
```

**実装方針**:
- `SSL_PRESETS` のキーをDropdownの選択肢として動的生成
- SSLモデル変更時に `VC.hubert_model` を再ロード（`VC.get_vc()` 内の既存のモデルキャッシュ管理に組み込み）
- 学習タブでは `preprocess_dataset()` 内の特徴量抽出コマンドに `--ssl_model` 引数として渡す
- SSLモデル選択状態は実験ディレクトリの `config.json` に `ssl_model` キーとして保存（どのSSLモデルで特徴量抽出したかをトレーサビリティ確保）

**影響範囲**:
- `infer-web.py`: Gradio UIの構築部分に Dropdown 追加、`vc_single()` / `vc_multi()` の引数にSSLモデル情報を追加
- `infer/modules/vc/modules.py`: `VC.get_vc()` と `VC.vc_single()` にSSLモデル設定を伝搬

---

### 2-6c. Weighted Sum of Layers 導入（2日）

> **注意**: milestones.md のM4候補タスク（4-2）にも同名タスクが残っているが、M2-A（2-6c）に前倒しされた。milestones.md の4-2は削除すべき重複エントリ。

**目的**: SSLモデルの特定レイヤー1つだけでなく、複数レイヤーの加重和を特徴量として使用し、表現力を向上させる。

**変更対象ファイル**:
- `infer/modules/vc/pipeline.py`
- `infer/modules/vc/ssl_wrapper.py`（新規モジュール内に実装）
- `infer/lib/infer_pack/models.py`

**設計方針**:

```python
# infer/modules/vc/ssl_wrapper.py に追加

class WeightedSumOfLayers(nn.Module):
    """
    SSLモデルの複数レイヤー出力の学習可能な加重和。
    HuBERT の layer 1~12 の出力を重み付き合成して1つの特徴量にする。
    
    参考: WavLM / data2vec の手法と同様のアプローチ。
    """
    def __init__(self, num_layers: int = 12):
        super().__init__()
        # 学習可能な重みパラメータ（softmax前）
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))
    
    def forward(self, hidden_states: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            hidden_states: list of (batch, time, hidden_dim) tensors
                           len = num_layers
        Returns:
            weighted_sum: (batch, time, hidden_dim) tensor
        """
        weights = torch.softmax(self.layer_weights, dim=0)
        stacked = torch.stack(hidden_states, dim=0)  # (num_layers, batch, time, dim)
        weighted_sum = (stacked * weights.view(-1, 1, 1, 1)).sum(dim=0)
        return weighted_sum
```

**パイプラインへの組み込み**:

`TransformersHuBERTWrapper` に `use_weighted_sum` フラグを追加:
```python
class TransformersHuBERTWrapper(nn.Module):
    def __init__(self, model_name_or_path, device="cpu", use_weighted_sum=False):
        super().__init__()
        self.model = HubertModel.from_pretrained(
            model_name_or_path, output_hidden_states=True
        )
        self.use_weighted_sum = use_weighted_sum
        if use_weighted_sum:
            config = AutoConfig.from_pretrained(model_name_or_path)
            self.weighted_sum = WeightedSumOfLayers(
                num_layers=config.num_hidden_layers
            )
    
    def extract_features(self, source, padding_mask, output_layer=12):
        attention_mask = ~padding_mask if padding_mask is not None else None
        outputs = self.model(
            input_values=source,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        if self.use_weighted_sum:
            # hidden_states[1:] = Transformer layer 1~12
            features = self.weighted_sum(
                list(outputs.hidden_states[1:output_layer + 1])
            )
        else:
            features = outputs.hidden_states[output_layer]
        return (features, padding_mask)
```

**注意事項**:
- `WeightedSumOfLayers` の重みは学習可能パラメータだが、RVCの fine-tuning 中に同時に最適化する設計。事前学習フェーズ（M3）で重みを学習し、推論時は固定
- 推論時に重みが未学習（デフォルト＝均等重み）でも動作するようにする（均等重みは layer 12 単独より性能が落ちる可能性があるため、デフォルトは `use_weighted_sum=False`）
- チェックポイントに `weighted_sum.layer_weights` を保存する設計

---

## 3. エージェントチーム構成

### 推奨ロール分担

| ロール | 担当サブタスク | スキル要件 |
|--------|---------------|-----------|
| **バックエンドエンジニアA** | 2-1, 2-2, 2-3 | PyTorch, transformers, fairseq の知識。SSLモデルの内部構造理解 |
| **バックエンドエンジニアB** | 2-4, 2-5 | RVCの学習パイプライン理解、チェックポイント互換性の実装経験 |
| **フロントエンド / 統合** | 2-6b | Gradio の知識、`infer-web.py` の構造理解 |
| **ML検証** | 2-6, 2-6c | 評価スクリプトの実行、CER/MCD の解釈、Weighted Sum の実装 |

### 作業順序の依存関係

```
2-1 (SSLローダー抽象化)
 ├── 2-2 (transformers ラッパー)
 │    ├── 2-3 (推論パイプライン対応) ─── 2-6 (検証)
 │    └── 2-4 (学習パイプライン対応)
 ├── 2-5 (ssl_dim パラメータ化) ※2-1と並行可能
 ├── 2-6b (WebUI) ※2-1, 2-3 完了後
 └── 2-6c (Weighted Sum) ※2-2 完了後、2-6と並行可能
```

**クリティカルパス**: 2-1 → 2-2 → 2-3 → 2-6（4.5日）

---

## 4. 提供範囲・テスト項目

### ユニットテスト

| テストID | テスト内容 | 対象ファイル | 合格基準 |
|----------|-----------|-------------|---------|
| UT-2-1a | `SSLModelConfig` のプリセットが正しく定義されていること | `utils.py` | 全プリセットの `ssl_dim`, `output_layer` が期待値と一致 |
| UT-2-1b | `load_ssl_model()` が fairseq backend でモデルをロードできること | `utils.py` | `assets/hubert/hubert_base.pt` をロードして `extract_features()` が呼べる |
| UT-2-1c | `load_ssl_model()` が transformers backend でモデルをロードできること | `utils.py` | HuggingFace モデルをロードして `extract_features()` が呼べる |
| UT-2-2a | `TransformersHuBERTWrapper` の出力形状が正しいこと | `ssl_wrapper.py` | ランダム入力 `(1, 16000)` に対して出力が `(1, T, 768)` |
| UT-2-2b | fairseq版とtransformers版のContentVec出力が一致すること | `ssl_wrapper.py` | cosine similarity > 0.99（同一音声入力時） |
| UT-2-2c | fp16推論が正常動作すること | `ssl_wrapper.py` | fp16入力でNaN/Infが発生しない |
| UT-2-5a | `ssl_dim=768` 指定時のモデル構築が既存と同一であること | `models.py` | パラメータ数が変更前後で一致 |
| UT-2-5b | 既存チェックポイント（`ssl_dim` 未保存）のロードが成功すること | `models.py` | `ssl_dim` が768にフォールバックされる |
| UT-2-5c | `modules.py` のモデルキャッシュクリアパス（L55-61, sid空文字列時）で `ssl_dim` が正しく伝搬されること | `modules.py` | エラーなくクリア処理が完了する |
| UT-2-6c | `WeightedSumOfLayers` の出力形状が正しいこと | `ssl_wrapper.py` | 入力12層 x `(1, T, 768)` → 出力 `(1, T, 768)` |
| UT-2-6c2 | `WeightedSumOfLayers` の重みが学習可能であること | `ssl_wrapper.py` | `requires_grad=True`、backward()でgrad計算可能 |

### E2Eテスト

| テストID | テスト内容 | 合格基準 |
|----------|-----------|---------|
| E2E-2a | ContentVec で推論した結果がM1ベースラインと同一であること | 既存モデルで推論、出力音声がビット完全一致（リファクタリングによる退行なし） |
| E2E-2b | kushinada で推論が正常完了すること | エラーなく音声ファイルが出力される |
| E2E-2c | ContentVec で特徴量抽出→学習→推論の全パイプラインが通ること | 学習が5 epoch 以上進行し、推論結果が得られる |
| E2E-2d | WebUIからSSLモデル切り替えが動作すること | Dropdown切り替え後の推論がエラーなく完了 |
| E2E-2e | リアルタイム変換（`rtrvc.py`）がSSLモデル切り替えに対応していること | リアルタイムGUIでの推論がエラーなく動作 |
| E2E-2f | DMLパッチが transformers 移行後も正常に適用されること | `--dml` フラグ付きでWebUI起動し、推論がエラーなく完了（DML環境がない場合はパッチ適用のコードパスが例外を出さないことを確認） |

### 性能テスト

| テストID | テスト内容 | 合格基準 |
|----------|-----------|---------|
| PERF-2a | transformers版の推論速度がfairseq版と同等以下であること | RTF（Real-Time Factor）差が20%以内 |
| PERF-2b | transformers版のVRAM使用量がfairseq版と同等以下であること | 差が500MB以内（RTX 4070 Ti Super 16GBで測定） |
| PERF-2c | Weighted Sum有効時の推論速度オーバーヘッド | RTF増加が10%以内 |

---

## 5. 懸念事項とレビュー項目

### 致命的リスク

#### R1: kushinadaが歌声で効果不足
- **詳細**: kushinada-hubert-base は62,215時間の日本語TV放送（話声）で学習されている。歌声は話声とドメインが異なり（ピッチ範囲、ビブラート、メリスマ等）、特徴量の質が話声ほど高くない可能性がある
- **影響**: M2-Aの主目的であるCER改善が達成できない
- **緩和策**: 
  - M1期間中に予備実験を実施（短い歌声サンプルでkushinadaの特徴量品質をラフチェック）
  - Spin V2（歌声含むデータで学習）を比較候補に追加
  - 最悪ケースではMERT（音楽特化SSL、ただし768次元ではない可能性）を検討
- **Go/No-Go判定**: 2-6の検証結果でCER改善なしの場合、M2-A完了としてSSLモデルローダー抽象化の成果のみをマージし、SSLモデル選択はM3に先送り

#### R2: fairseq依存の非互換
- **詳細**: 現在のfairseqは `github.com/One-sixth/fairseq.git` のフォークを使用。PyTorch 2.10.0 との互換性は確認済みだが、transformers移行中に fairseq を完全に外すと DML（DirectML）パッチ（`fairseq.modules.grad_multiply.GradMultiply.forward`のモンキーパッチ）が動作しなくなる
- **影響**: DML環境（AMD GPU等）でのリアルタイム変換が動作しなくなる
- **緩和策**: fairseqはpyproject.tomlから外さず、DMLパッチは条件分岐で fairseq 利用を維持。transformers移行は推論・学習のSSLモデルロード部分のみ対象とする

### 高リスク

#### R3: spk_embed_dim変更の互換性
- **詳細**: `configs/v2/48k.json` の `"spk_embed_dim": 109` はモデル定義の `emb_g` サイズを決定する。事前学習済みチェックポイントはこの値で学習されているため、変更するとロード不可になる
- **緩和策**: 既存config（48k.json, 32k.json）は変更しない。事前学習用configは `configs/v2/48k_pretrain.json` として別管理

#### R4: 特徴量ディレクトリ名の互換性
- **詳細**: 特徴量は `logs/{実験名}/3_feature768/` に保存される。SSLモデル変更時にこのディレクトリ名を変更すると、既存の学習済み実験の filelist.txt が壊れる
- **緩和策**: `ssl_dim=768`（今回のターゲットSSLモデルはすべて768次元）の場合はディレクトリ名を変更しない。将来的に非768次元のSSLモデルに対応する際は `3_feature{ssl_dim}` とする

#### R5: transformers ライブラリのバージョン競合
- **詳細**: transformers は依存が重く、既存の `fairseq`, `torch`, `onnxruntime-gpu` との依存解決が困難になる可能性がある
- **緩和策**: `uv sync` で依存解決を確認。解決不可の場合は `transformers` のSSLモデル部分のみを手動で切り出す（`HubertModel` のソースコードをvendoring）

### レビューチェックリスト

- [ ] `load_ssl_model()` が `config.device` と `config.is_half` を正しく反映しているか
- [ ] transformers版ラッパーの `extract_features()` が fairseq 版と同じシグネチャ・戻り値型を返すか
- [ ] `output_layer` のハードコード箇所が全て設定値に置換されているか（grep `"output_layer": 12` で0件になること）
- [ ] 既存の `.pth` チェックポイント（`ssl_dim` キーなし）がエラーなくロードできるか
- [ ] `WeightedSumOfLayers` の重みが `state_dict()` に含まれ、`load_state_dict()` で復元可能か
- [ ] DMLパッチ（`GradMultiply.forward` モンキーパッチ）が transformers 移行後も動作するか
- [ ] WebUI Dropdown の選択肢がSSL_PRESETSと同期しているか
- [ ] `infer-web.py` の `fea_dim = 768` ハードコードが `ssl_config.ssl_dim` に置換されているか
- [ ] `faiss.index_factory(768, ...)` のハードコードが `ssl_config.ssl_dim` に置換されているか
- [ ] `extract_feature_print.py` の `outPath = "%s/3_feature768"` が動的化されているか

---

## 6. 一から作り直すとしたら（M2フェーズ全体の理想設計）

もしRVCのSSLモデル統合を白紙の状態から設計するとしたら、以下のアーキテクチャが理想的:

### SSLモデルレジストリ

```python
# infer/modules/ssl/registry.py

class SSLModelRegistry:
    """
    SSLモデルのプラグイン式レジストリ。
    新しいSSLモデルを追加する際はここにエントリを追加するだけ。
    """
    _models: dict[str, type[BaseSSLModel]] = {}
    
    @classmethod
    def register(cls, name: str):
        def decorator(model_cls):
            cls._models[name] = model_cls
            return model_cls
        return decorator
    
    @classmethod
    def create(cls, name: str, **kwargs) -> BaseSSLModel:
        return cls._models[name](**kwargs)

@SSLModelRegistry.register("contentvec")
class ContentVecModel(BaseSSLModel): ...

@SSLModelRegistry.register("kushinada")
class KushinadaModel(BaseSSLModel): ...
```

### 設定の一元管理

現在のRVCは設定が複数箇所に散在している（`configs/v2/*.json`, `.env`, `config.py`, コマンドライン引数, チェックポイント内の `cpt["config"]` リスト）。理想的には:

```yaml
# configs/experiment.yaml
ssl:
  model: kushinada
  output_layer: 12
  weighted_sum: true
  weighted_sum_layers: [6, 7, 8, 9, 10, 11, 12]

model:
  ssl_dim: 768  # ssl.model から自動推論
  inter_channels: 192
  hidden_channels: 192
  # ...

training:
  pretrained_g: assets/pretrained_v2/f0G48k.pth
  pretrained_d: assets/pretrained_v2/f0D48k.pth
  # ...
```

### SynthesizerTrn の統合

現在は `SynthesizerTrnMs768NSFsid`（F0あり）と `SynthesizerTrnMs768NSFsid_nono`（F0なし）が別クラスだが、これは `f0=True/False` フラグで統一すべき:

```python
class SynthesizerTrn(nn.Module):
    def __init__(self, config: ModelConfig):
        # ssl_dim, f0, sr 等はすべて config から取得
        self.enc_p = TextEncoder(config.ssl_dim, ...)
        if config.f0:
            self.dec = GeneratorNSF(...)
        else:
            self.dec = Generator(...)
```

### 段階的移行が現実的

ただし、既存チェックポイントの互換性とコミュニティの利用状況を考えると、上記の理想設計への一括移行は現実的ではない。M2-AではSSLモデルローダーの抽象化と `ssl_dim` パラメータ化にとどめ、設定の一元化やクラス統合はM3以降で段階的に実施する。

---

## 7. 後続タスクへの連絡事項

### M2-B（Phase 2-B: 日本語歌声事前学習）への連絡

- `SSLModelConfig` の `ssl_dim` と `output_layer` は `extract_feature_print.py` を通じて学習パイプラインに伝搬される。M2-Bで事前学習を行う際は、SSLモデルの切り替えに伴う特徴量キャッシュの無効化ロジックに注意
- `3_feature768` ディレクトリの命名規則はM2-Aでは変更しないが、M2-Bでは `3_feature_{ssl_model_name}` のようなSSLモデル名を含むディレクトリ構造を検討してもよい
- M2-Aで構築した `load_ssl_model()` と `TransformersHuBERTWrapper` はM2-Bの事前学習スクリプトでそのまま利用可能
- kushinada で事前学習する場合、`configs/v2/48k_pretrain.json` に `ssl_model: "kushinada"` を追記する設計を想定。このファイルはM2-Aでテンプレートを作成しておく
- `WeightedSumOfLayers` の重みパラメータは事前学習フェーズ（M2-B）で初めて学習される。M2-A段階ではデフォルト（均等重み）のまま

### M3（損失関数+ボコーダ改善）への連絡

- M3-Bでボコーダ改善（SnakeBeta等）を行う際に事前学習の再実行が必要になるが、その際もM2-Aで構築したSSLモデルローダーをそのまま利用する
- M3-B事前学習再実行時に `mel_fmin=40.0` 変更も同時適用する設計（milestones M3-B参照）

### チェックポイント形式の変更

M2-A以降で保存されるチェックポイントには以下のキーが追加される:
- `ssl_dim`: int（SSLモデルの出力次元）
- `ssl_model`: str（使用したSSLモデルのプリセット名）
- `ssl_output_layer`: int（使用したレイヤー番号）

これらは旧チェックポイントには存在しないため、ロード時に `cpt.get("ssl_dim", 768)` のようにデフォルト値でフォールバックする。

### fairseq完全削除のタイムライン

M2-Aでは fairseq を削除しない。削除の前提条件:
1. DMLパッチの代替策確立（`GradMultiply.forward` モンキーパッチの代替）
2. `infer/lib/jit/get_hubert.py` の fairseq 依存解消（JITコンパイル用のHuBERTロード）
3. 全ユーザーのチェックポイントが transformers 版で問題なく動作することの確認

目標はM4（最終統合）での完全削除。

### `infer-web.py` UI変更のサマリ

推論タブと学習タブにSSLモデル選択Dropdownを追加する。これにより `infer-web.py` の引数伝搬が変更されるため、API (`api_240604.py`) にも同様のパラメータ追加が必要になる可能性がある。M2-AではWebUIのみ対応し、API対応はM2-B以降とする。
