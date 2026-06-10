# RVC Deep Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Break up the three monolithic files (`infer-web.py` 1619 lines, `infer/lib/infer_pack/models.py` 1223 lines, `gui_v1.py` 1070 lines), de-duplicate the realtime stack, and land verified-safe performance fixes — without changing any public behavior, class name, or import path that other code or checkpoints depend on.

**Architecture:** Pure mechanical decomposition with backwards-compatible re-exports. `models.py` keeps the 4 `SynthesizerTrn*` classes (their names are hardcoded in 6 files for checkpoint instantiation) and re-exports everything it loses, so every existing `from infer.lib.infer_pack.models import X` keeps working. The WebUI moves into a new `web/` package with `infer-web.py` as a thin launcher. Realtime duplication (phase_vocoder, Harvest, device queries — verbatim identical between `gui_v1.py` and `tools/realtime/engine.py`) collapses into shared modules under `tools/realtime/`.

**Tech Stack:** Python 3.9 venv at `.venv/` (torch, gradio, faiss, sounddevice all importable). No pytest — verification is plain-python scripts under `tests/` plus `tools/realtime/test_protocol.py`.

**Verification baseline (every task ends with):**
```bash
.venv/bin/python tests/smoke_imports.py        # must print ALL OK
.venv/bin/python -m py_compile infer-web.py gui_v1.py
.venv/bin/black --check <changed files>
git commit
```

**Hard constraints (do not violate):**
- Class names `SynthesizerTrnMs256NSFsid`, `SynthesizerTrnMs768NSFsid`, `SynthesizerTrnMs256NSFsid_nono`, `SynthesizerTrnMs768NSFsid_nono` must stay importable from `infer.lib.infer_pack.models` (used by `infer/modules/vc/modules.py:110-113`, `infer/lib/jit/get_synthesizer.py`, `infer/modules/train/train.py:59-66`, `tools/rvc_for_realtime.py`, `tools/infer/infer-pm-index256.py`).
- `tools/rvc_for_realtime.py` is used by `api_240604.py` / `api_231006.py` — do not modify.
- `pyworld.harvest` requires float64 input — do NOT change `astype(np.double)` calls.
- `audio_pad[i : i - self.window]` in pipeline.py is a correct negative-index sliding-window idiom, not a bug.
- `models_onnx.py` / `attentions_onnx.py` stay as-is (ONNX tracing constraints make merging risky).

---

### Task 0: Test harness + baseline

**Files:**
- Create: `tests/smoke_imports.py`
- Create: `tests/model_construct.py`

- [ ] **Step 1: Write smoke import script**

```python
# tests/smoke_imports.py
"""Import every module the refactor touches. Run from repo root: .venv/bin/python tests/smoke_imports.py"""
import os, sys

sys.path.insert(0, os.getcwd())

MODULES = [
    "configs.config",
    "infer.lib.infer_pack.models",
    "infer.lib.infer_pack.models_onnx",
    "infer.lib.infer_pack.attentions",
    "infer.modules.vc.modules",
    "infer.modules.vc.pipeline",
    "infer.lib.rtrvc",
    "infer.lib.rmvpe",
    "infer.lib.train.process_ckpt",
    "tools.realtime.engine",
    "tools.realtime.harvest_worker",
    "i18n.i18n",
]

import importlib

for m in MODULES:
    importlib.import_module(m)
    print("ok", m)
print("ALL OK")
```

- [ ] **Step 2: Write model-construction fingerprint script**

```python
# tests/model_construct.py
"""Construct all synthesizers + discriminators from real configs and print a
state_dict fingerprint. Run before and after the models.py split; output must be identical."""
import os, sys, json, hashlib

sys.path.insert(0, os.getcwd())
import torch

from infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
    MultiPeriodDiscriminator,
    MultiPeriodDiscriminatorV2,
)

torch.manual_seed(0)


def fingerprint(model):
    keys = sorted(model.state_dict().keys())
    h = hashlib.sha256("\n".join(keys).encode()).hexdigest()[:16]
    return len(keys), h


with open("configs/v1/40k.json") as f:
    cfg = json.load(f)

kw = dict(
    spec_channels=cfg["data"]["filter_length"] // 2 + 1,
    segment_size=cfg["train"]["segment_size"] // cfg["data"]["hop_length"],
    **cfg["model"],
    sr=cfg["data"]["sampling_rate"],
)
for cls in (SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono):
    print(cls.__name__, *fingerprint(cls(**kw)))

with open("configs/v2/48k.json") as f:
    cfg = json.load(f)
kw = dict(
    spec_channels=cfg["data"]["filter_length"] // 2 + 1,
    segment_size=cfg["train"]["segment_size"] // cfg["data"]["hop_length"],
    **cfg["model"],
    sr=cfg["data"]["sampling_rate"],
)
for cls in (SynthesizerTrnMs768NSFsid, SynthesizerTrnMs768NSFsid_nono):
    print(cls.__name__, *fingerprint(cls(**kw)))

print("MPD", *fingerprint(MultiPeriodDiscriminator(False)))
print("MPDv2", *fingerprint(MultiPeriodDiscriminatorV2(False)))
print("DONE")
```

Note: synthesizer `__init__` signatures take positional args `(spec_channels, segment_size, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, spk_embed_dim, gin_channels, sr, ...)` — check `infer/modules/train/train.py:59-66` for exactly how the config maps if the kwargs above don't line up; adjust the script (not the model) until it runs on the **unmodified** codebase.

- [ ] **Step 3: Run both scripts on the unmodified tree; save baseline**

Run: `.venv/bin/python tests/smoke_imports.py` → expect `ALL OK`
Run: `.venv/bin/python tests/model_construct.py > /tmp/model_fingerprint_before.txt && cat /tmp/model_fingerprint_before.txt` → expect `DONE` with 6 fingerprint lines

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "test: add smoke-import and model-fingerprint harness for refactor"
```

---

### Task 1: Split `infer/lib/infer_pack/models.py`

**Files:**
- Create: `infer/lib/infer_pack/encoders.py` (TextEncoder lines 19-80, ResidualCouplingBlock 82-146, PosteriorEncoder 148-202)
- Create: `infer/lib/infer_pack/generators.py` (Generator 204-310, SineGen 312-389, SourceModuleHnNSF 391-446, GeneratorNSF 448-593)
- Create: `infer/lib/infer_pack/discriminators.py` (MultiPeriodDiscriminator 1052-1080, MultiPeriodDiscriminatorV2 1082-1110, DiscriminatorS 1112-1140, DiscriminatorP 1142-1223)
- Modify: `infer/lib/infer_pack/models.py` (keeps 4 synthesizers + re-exports)

- [ ] **Step 1: Create the three new modules**

Each new file starts with the subset of models.py's current import header (lines 1-17) that its classes actually use, then the class bodies moved **verbatim**. models.py current header for reference:

```python
import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)

import numpy as np
import torch
from torch import nn
from torch.nn import AutogradContext  # <- copy whatever is actually there; do not invent
```

(Read the real lines 1-17 first; copy exactly, trimming unused names per file. `encoders.py` needs `commons`, `modules`, `attentions` imports; `generators.py` needs `modules`, `commons`, `Conv1d/ConvTranspose1d/remove_weight_norm/weight_norm` etc.; `discriminators.py` needs `Conv1d, Conv2d, spectral_norm, weight_norm`, `commons.get_padding`, `modules`.)

- [ ] **Step 2: Rewrite models.py**

Delete the moved class bodies. At the top (after the existing import header, which keeps whatever the synthesizers still use), add:

```python
from infer.lib.infer_pack.encoders import (
    TextEncoder,
    ResidualCouplingBlock,
    PosteriorEncoder,
)
from infer.lib.infer_pack.generators import (
    Generator,
    SineGen,
    SourceModuleHnNSF,
    GeneratorNSF,
)
from infer.lib.infer_pack.discriminators import (
    MultiPeriodDiscriminator,
    MultiPeriodDiscriminatorV2,
    DiscriminatorS,
    DiscriminatorP,
)
```

Keep the 4 SynthesizerTrn* classes in models.py unchanged.

- [ ] **Step 3: Verify fingerprints unchanged**

Run: `.venv/bin/python tests/model_construct.py > /tmp/model_fingerprint_after.txt && diff /tmp/model_fingerprint_before.txt /tmp/model_fingerprint_after.txt`
Expected: no diff output (exit 0)
Run: `.venv/bin/python tests/smoke_imports.py` → `ALL OK`

- [ ] **Step 4: Format and commit**

```bash
.venv/bin/black infer/lib/infer_pack/
git add infer/lib/infer_pack/
git commit -m "refactor: split infer_pack/models.py into encoders/generators/discriminators"
```

---

### Task 2: Split `infer-web.py` into a `web/` package

**Files:**
- Create: `web/__init__.py` (empty)
- Create: `web/runtime.py` (infer-web.py lines 1-178: env setup, logging, config, vc, i18n, GPU detection, weight roots, names/index_paths/uvr5_names, `lookup_indices`, `change_choices`, `clean`, `ToolButton`, `F0GPUVisible`)
- Create: `web/train_ops.py` (lines 181-806: `export_onnx`, `if_done`, `if_done_multi`, `preprocess_dataset`, `extract_f0_feature`, `get_pretrained_models`, `change_sr2`, `change_version19`, `change_f0`, `click_train`, `train_index`, `train1key`, `change_info_`, `change_f0_method`)
- Create: `web/tabs/__init__.py`, `web/tabs/inference.py`, `web/tabs/uvr5.py`, `web/tabs/train.py`, `web/tabs/ckpt.py`, `web/tabs/onnx.py`, `web/tabs/faq.py`
- Create: `web/app.py` (`build_app()` assembling the Blocks)
- Rewrite: `infer-web.py` (thin launcher)
- Create: `tests/webui_build.py`

- [ ] **Step 1: Create `web/runtime.py`**

Move infer-web.py lines 1-178 verbatim (minus imports only used by training ops: `faiss`, `MiniBatchKMeans`, `Popen`, `shuffle`, `threading`, `pathlib`, `json` move to train_ops). Keep: dotenv load, `now_dir`/sys.path, tmp/TEMP setup, torch seed, logging config, the fairseq DML override block (lines 57-64), `config = Config()`, `vc = VC(config)`, `i18n = I18nAuto()`, GPU detection block producing `gpu_info`/`default_batch_size`/`gpus`, `weight_root`/`weight_uvr5_root`/`index_root`/`outside_index_root`, `names`/`index_paths`/`uvr5_names` listing, `lookup_indices`, `change_choices`, `clean`, `ToolButton`, `F0GPUVisible = config.dml == False`.

- [ ] **Step 2: Create `web/train_ops.py`**

Header:

```python
import json
import logging
import os
import pathlib
import platform
import shutil
import threading
import traceback
from random import shuffle
from subprocess import Popen
from time import sleep

import numpy as np
import torch

from configs.config import Config
from web.runtime import config, i18n, now_dir, outside_index_root, F0GPUVisible

logger = logging.getLogger(__name__)
```

Move the 14 functions verbatim. Inside `train_index` and `train1key`, add lazy imports at the top of the function body (startup-time win — faiss and sklearn were previously imported even when only doing inference):

```python
    import faiss
    from sklearn.cluster import MiniBatchKMeans
```

- [ ] **Step 3: Create tab builders**

Each tab module exposes `def build():` and is called inside the `gr.Blocks` context (gradio tracks context globally, so plain function calls work). Move the UI code for each tab verbatim, indented under `def build():`:

- `web/tabs/inference.py` — lines 817-1112 (the whole 模型推理 TabItem including both sub-tabs and the `sid0.change` wiring). Imports: `gradio as gr`, `from web.runtime import vc, i18n, names, index_paths, change_choices, clean, ToolButton`.
- `web/tabs/uvr5.py` — lines 1113-1170. Imports: `gradio as gr`, `from infer.modules.uvr5.modules import uvr`, `from web.runtime import i18n, uvr5_names`.
- `web/tabs/train.py` — lines 1171-1422. Imports: `gradio as gr`, `from web.runtime import i18n, gpu_info, gpus, default_batch_size, change_choices, F0GPUVisible`, `from web.train_ops import (...)`.
- `web/tabs/ckpt.py` — lines 1424-1579. Imports: `gradio as gr`, `from infer.lib.train.process_ckpt import change_info, extract_small_model, merge, show_info`, `from web.runtime import i18n`, `from web.train_ops import change_info_`.
- `web/tabs/onnx.py` — lines 1581-1596. Imports: `gradio as gr`, `from web.runtime import i18n`, `from web.train_ops import export_onnx`.
- `web/tabs/faq.py` — lines 1598-1609. Imports: `gradio as gr`, `from web.runtime import i18n`.

- [ ] **Step 4: Create `web/app.py` and rewrite `infer-web.py`**

```python
# web/app.py
import gradio as gr

from web.runtime import i18n
from web.tabs import ckpt, faq, inference, onnx, train, uvr5


def build_app() -> gr.Blocks:
    with gr.Blocks(title="RVC WebUI") as app:
        gr.Markdown("## RVC WebUI")
        gr.Markdown(
            value=i18n(
                "本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>LICENSE</b>."
            )
        )
        with gr.Tabs():
            inference.build()
            uvr5.build()
            train.build()
            ckpt.build()
            onnx.build()
            faq.build()
    return app
```

(Copy the exact Markdown header lines from current infer-web.py 810-816 — do not retype from this plan.)

```python
# infer-web.py
import os
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)

from configs.config import Config  # noqa: E402  (ensures CLI args parsed first if needed)
from web.app import build_app  # noqa: E402
from web.runtime import config  # noqa: E402

app = build_app()

if config.iscolab:
    app.queue(concurrency_count=511, max_size=1022).launch(share=True)
else:
    app.queue(concurrency_count=511, max_size=1022).launch(
        server_name="0.0.0.0",
        inbrowser=not config.noautoopen,
        server_port=config.listen_port,
        quiet=True,
    )
```

(Copy the exact launch block from current lines 1611-1619 — match args exactly, including `favicon_path` etc. if present.)

- [ ] **Step 5: Build test**

```python
# tests/webui_build.py
"""Builds the full Gradio app without launching. Run: .venv/bin/python tests/webui_build.py"""
import os, sys

sys.path.insert(0, os.getcwd())

from web.app import build_app

app = build_app()
assert app is not None
print("blocks:", len(app.blocks))
print("WEBUI BUILD OK")
```

Run: `.venv/bin/python tests/webui_build.py` → expect `WEBUI BUILD OK` and a block count > 200.

- [ ] **Step 6: Format, verify, commit**

```bash
.venv/bin/black web/ infer-web.py tests/webui_build.py
.venv/bin/python tests/smoke_imports.py
.venv/bin/python tests/webui_build.py
git add web/ infer-web.py tests/webui_build.py
git commit -m "refactor: split infer-web.py into web/ package (runtime, train_ops, tab modules)"
```

---

### Task 3: De-duplicate the realtime stack

**Files:**
- Create: `tools/realtime/dsp.py` (shared `phase_vocoder`)
- Create: `tools/realtime/devices.py` (shared device-query functions)
- Modify: `gui_v1.py` (drop local `Harvest` 50-72, `phase_vocoder` 26-47, device methods 1010-1068 → import/delegate)
- Modify: `tools/realtime/engine.py` (drop local `phase_vocoder` 36-57, device methods 127-180 → import/delegate)

- [ ] **Step 1: Create `tools/realtime/dsp.py`** — move `phase_vocoder` verbatim from engine.py lines 36-57 (identical to gui_v1.py 26-47), with `import torch` header.

- [ ] **Step 2: Create `tools/realtime/devices.py`** — extract the verbatim device-query logic into stateless functions:

```python
"""Shared sounddevice device-query helpers used by gui_v1 and the realtime engine."""
import sounddevice as sd


def query_devices(hostapi_name=None):
    """Returns (hostapis, input_devices, output_devices, input_devices_indices, output_devices_indices)."""
    sd._terminate()
    sd._initialize()
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    for hostapi in hostapis:
        for device_idx in hostapi["devices"]:
            devices[device_idx]["hostapi_name"] = hostapi["name"]
    hostapi_names = [hostapi["name"] for hostapi in hostapis]
    if hostapi_name not in hostapi_names:
        hostapi_name = hostapi_names[0]
    input_devices = [
        d["name"] for d in devices
        if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
    ]
    output_devices = [
        d["name"] for d in devices
        if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name
    ]
    input_devices_indices = [
        d["index"] if "index" in d else d["name"] for d in devices
        if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
    ]
    output_devices_indices = [
        d["index"] if "index" in d else d["name"] for d in devices
        if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name
    ]
    return hostapi_names, input_devices, output_devices, input_devices_indices, output_devices_indices
```

**IMPORTANT:** Before writing this, read the real `update_devices` bodies in both files (gui_v1.py 1010-1043, engine.py 127-158) and copy their exact logic — the snippet above is the expected shape, but the source of truth is the existing code. Also move `get_device_samplerate()` and `get_device_channels()` logic into module functions `get_device_samplerate()` / `get_device_channels(config_or_none)` matching the existing bodies exactly.

- [ ] **Step 3: Rewire both consumers.** In each file, replace the method bodies with thin delegations that keep the same attribute side-effects, e.g. in engine.py:

```python
    def update_devices(self, hostapi_name=None):
        (
            self.hostapis,
            self.input_devices,
            self.output_devices,
            self.input_devices_indices,
            self.output_devices_indices,
        ) = devices.query_devices(hostapi_name)
```

In gui_v1.py also: delete the `Harvest` class and add `from tools.realtime.harvest_worker import Harvest`; delete `phase_vocoder` and add `from tools.realtime.dsp import phase_vocoder`; same import in engine.py.

- [ ] **Step 4: Verify**

```bash
.venv/bin/python -m py_compile gui_v1.py tools/realtime/engine.py tools/realtime/devices.py tools/realtime/dsp.py
.venv/bin/python tests/smoke_imports.py
.venv/bin/python tools/realtime/test_protocol.py   # existing headless protocol test
```
Expected: protocol test prints its pass output (read its tail to learn the exact success marker before running).

- [ ] **Step 5: Format and commit**

```bash
.venv/bin/black gui_v1.py tools/realtime/
git add gui_v1.py tools/realtime/
git commit -m "refactor: extract shared realtime dsp/device helpers, dedupe gui_v1 and engine"
```

---

### Task 4: Performance fixes (verified-safe only)

**Files:**
- Modify: `infer/modules/vc/pipeline.py` (faiss index cache; vectorized silence search)
- Modify: `infer/lib/rtrvc.py` (index reload guard)
- Create: `tests/pipeline_equiv.py`

- [ ] **Step 1: Write the equivalence test FIRST**

```python
# tests/pipeline_equiv.py
"""Proves the vectorized silence-point search matches the original loop."""
import numpy as np

WINDOW = 160
T_QUERY = 16000 * 6
T_CENTER = 16000 * 38


def original(audio, audio_pad, window, t_center, t_query):
    audio_sum = np.zeros_like(audio)
    for i in range(window):
        audio_sum += np.abs(audio_pad[i : i - window])
    opt_ts = []
    for t in range(t_center, audio.shape[0], t_center):
        opt_ts.append(
            t - t_query
            + np.where(
                audio_sum[t - t_query : t + t_query]
                == audio_sum[t - t_query : t + t_query].min()
            )[0][0]
        )
    return opt_ts


def vectorized(audio, audio_pad, window, t_center, t_query):
    abs_pad = np.abs(audio_pad)
    csum = np.concatenate(([0.0], np.cumsum(abs_pad)))
    audio_sum = (csum[window:] - csum[:-window])[: audio.shape[0]]
    opt_ts = []
    for t in range(t_center, audio.shape[0], t_center):
        seg = audio_sum[t - t_query : t + t_query]
        opt_ts.append(t - t_query + int(np.argmin(seg)))
    return opt_ts


rng = np.random.default_rng(0)
for trial in range(5):
    n = int(16000 * (40 + 80 * rng.random()))  # 40-120s of 16k audio
    audio = rng.standard_normal(n).astype(np.float32) * 0.1
    # carve silent valleys so argmin targets are unambiguous
    for _ in range(8):
        c = rng.integers(T_QUERY, n - T_QUERY)
        audio[c - 400 : c + 400] *= 0.001
    audio_pad = np.pad(audio, (WINDOW // 2, WINDOW // 2), mode="reflect")
    a = original(audio, audio_pad, WINDOW, T_CENTER, T_QUERY)
    b = vectorized(audio, audio_pad, WINDOW, T_CENTER, T_QUERY)
    assert a == b, (trial, a, b)
    print("trial", trial, "ok", len(a), "split points")
print("PIPELINE EQUIV OK")
```

Run: `.venv/bin/python tests/pipeline_equiv.py` → `PIPELINE EQUIV OK`. If float-accumulation order flips an argmin in some trial, relax the assert to compare `audio_sum` values at the chosen indices within 1e-4 — the split point is a heuristic, but get the test green and honest before touching pipeline.py.

- [ ] **Step 2: Apply the vectorization in `pipeline.py` (lines 321-333)**

Replace:
```python
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += np.abs(audio_pad[i : i - self.window])
            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t
                    - self.t_query
                    + np.where(
                        audio_sum[t - self.t_query : t + self.t_query]
                        == audio_sum[t - self.t_query : t + self.t_query].min()
                    )[0][0]
                )
```
with:
```python
        if audio_pad.shape[0] > self.t_max:
            abs_pad = np.abs(audio_pad)
            csum = np.concatenate(([0.0], np.cumsum(abs_pad)))
            audio_sum = (csum[self.window :] - csum[: -self.window])[: audio.shape[0]]
            for t in range(self.t_center, audio.shape[0], self.t_center):
                seg = audio_sum[t - self.t_query : t + self.t_query]
                opt_ts.append(t - self.t_query + int(np.argmin(seg)))
```

- [ ] **Step 3: Add faiss index cache in `pipeline.py` (lines 302-317)**

Module level (near `input_audio_path2wav`):
```python
# One-entry cache: batch inference calls pipeline() once per file with the same
# index, and reading + reconstructing a large IVF index costs 100ms-1s per call.
_cached_index = {"key": None, "index": None, "big_npy": None}
```

Replace the read block:
```python
            try:
                index = faiss.read_index(file_index)
                # big_npy = np.load(file_big_npy)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except:
                traceback.print_exc()
                index = big_npy = None
```
with:
```python
            try:
                key = (file_index, os.path.getmtime(file_index))
                if _cached_index["key"] != key:
                    _cached_index["index"] = faiss.read_index(file_index)
                    _cached_index["big_npy"] = _cached_index["index"].reconstruct_n(
                        0, _cached_index["index"].ntotal
                    )
                    _cached_index["key"] = key
                index = _cached_index["index"]
                big_npy = _cached_index["big_npy"]
            except:
                traceback.print_exc()
                index = big_npy = None
```

- [ ] **Step 4: Guard index reload in `infer/lib/rtrvc.py`**

In `__init__`, the `if index_rate != 0:` block (lines 82-85) gains an else; and `change_index_rate` (198-203) becomes:

```python
            if index_rate != 0:
                self.index = faiss.read_index(index_path)
                self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
                printt("Index search enabled")
            else:
                self.index = None
                self.big_npy = None
```

```python
    def change_index_rate(self, new_index_rate):
        if new_index_rate != 0 and self.index_rate == 0:
            if self.index is None:
                self.index = faiss.read_index(self.index_path)
                self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
                printt("Index search enabled")
        self.index_rate = new_index_rate
```

Check first whether `self.index` is referenced anywhere expecting AttributeError-when-absent semantics (grep `\.index\b` in rtrvc.py and engine.py); `infer()` gates on `self.index_rate != 0` so a None attribute is safe.

- [ ] **Step 5: Verify, format, commit**

```bash
.venv/bin/python tests/pipeline_equiv.py
.venv/bin/python tests/smoke_imports.py
.venv/bin/black infer/modules/vc/pipeline.py infer/lib/rtrvc.py tests/pipeline_equiv.py
git add infer/modules/vc/pipeline.py infer/lib/rtrvc.py tests/pipeline_equiv.py
git commit -m "perf: cache faiss index reads, vectorize silence-point search, guard index reload"
```

---

### Task 5: Final sweep — black, full verification, docs

**Files:**
- Modify: `CLAUDE.md` (architecture section: new `web/` package, infer_pack split, realtime shared modules, tests/)

- [ ] **Step 1: Full-repo black check** — `.venv/bin/black --check . --exclude '\.venv|node_modules'`; fix anything flagged in files we touched.

- [ ] **Step 2: Run everything**

```bash
.venv/bin/python tests/smoke_imports.py
.venv/bin/python tests/model_construct.py | diff /tmp/model_fingerprint_before.txt -
.venv/bin/python tests/webui_build.py
.venv/bin/python tests/pipeline_equiv.py
.venv/bin/python tools/realtime/test_protocol.py
```
All must pass.

- [ ] **Step 3: Update CLAUDE.md** — under Architecture, document: `web/` package (runtime/train_ops/tabs/app), `infer/lib/infer_pack/` split files with the rule "SynthesizerTrn* class names must never be renamed (checkpoint instantiation)", `tools/realtime/{dsp,devices,harvest_worker}.py` shared modules, and `tests/` plain-python verification scripts.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document refactored layout in CLAUDE.md"
```
