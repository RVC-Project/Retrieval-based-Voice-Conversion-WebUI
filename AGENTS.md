# AGENTS.md

Guidance for AI coding agents working in this repository. See `CLAUDE.md` for the Claude Code–specific version of these instructions.

## Project Overview

RVC (Retrieval-based Voice Conversion) is a voice conversion framework built on VITS, with HuBERT feature extraction, FAISS retrieval indices, RMVPE/Crepe/Harvest F0 extractors, a Gradio WebUI, realtime voice conversion (~90–170ms latency), UVR5 vocal separation, and FastAPI helpers. ONNX export and multiple hardware backends (CUDA, DirectML, ROCm, IPEX) are supported.

Keep changes small and targeted. `CONTRIBUTING.md` notes that algorithm changes are generally rejected upstream unless they fix code-level bugs or warnings; UI, translations, and operational fixes are safer contribution areas.

## Entry Points

- `infer-web.py`: thin launcher for the Gradio WebUI (default port `7865`). The actual UI lives in the `web/` package:
  - `web/runtime.py`: one-time setup (env, Config/VC/i18n singletons, GPU detection, model+index discovery)
  - `web/train_ops.py`: training subprocess orchestration and Gradio callbacks
  - `web/tabs/{inference,uvr5,train,ckpt,onnx,faq}.py`: one module per tab, each exposing `build()`
  - `web/app.py`: `build_app()` assembles the Blocks (used headlessly by `tests/webui_build.py`)
- `gui_v1.py`: realtime desktop GUI (FreeSimpleGUI + sounddevice)
- `api_240604.py` / `api_231006.py`: FastAPI-based APIs (240604 is the latest)
- `tools/infer_cli.py`: command-line single-file conversion
- `tools/infer_batch_rvc.py`: batch processing
- `tools/export_onnx.py`: ONNX model export

## Architecture

- `infer/modules/vc/modules.py`: high-level `VC` class — model loading, inference, index selection; caches loaded models
- `infer/modules/vc/pipeline.py`: core conversion pipeline — F0 extraction, RMS mixing, FAISS index use
- `infer/modules/train/`: preprocessing, F0/feature extraction, distributed training
- `infer/modules/uvr5/` + `infer/lib/uvr5_pack/`: UVR5 separation
- `infer/lib/infer_pack/`: VITS-based synthesizer models, split across `encoders.py` (TextEncoder, PosteriorEncoder, ResidualCouplingBlock), `generators.py` (Generator, GeneratorNSF, SineGen, SourceModuleHnNSF), `discriminators.py`, and `models.py` (the four `SynthesizerTrn*` classes). `models.py` re-exports everything — always import from `infer.lib.infer_pack.models`.
- Realtime stack: `infer/lib/rtrvc.py` is the canonical realtime RVC engine (used by `gui_v1.py` and the Electron server); `tools/rvc_for_realtime.py` is a simplified variant kept for the API servers. Shared helpers live in `tools/realtime/`: `dsp.py` (phase_vocoder, printt), `devices.py` (sounddevice queries), `harvest_worker.py` (Harvest F0 process).

### Hard Constraints

- **Never rename the `SynthesizerTrnMs{256,768}NSFsid[_nono]` classes** — checkpoint loading instantiates them by these exact names in `infer/modules/vc/modules.py`, `infer/lib/jit/get_synthesizer.py`, and `infer/modules/train/train.py`.
- `models_onnx.py` / `attentions_onnx.py` are intentionally separate ONNX-traceable variants; do not merge them with the torch versions.
- Preserve existing i18n keys and locale JSON structure when editing UI text; many files contain Chinese comments/UI strings.

## Setup and Running

- Supported Python: 3.10–3.13 (DirectML limited to 3.10–3.12, IPEX to 3.10).
- Install PyTorch first for the target hardware, then the matching requirements file: `requirements.txt` (NVIDIA/general), `requirements-dml.txt`, `requirements-amd.txt`, `requirements-ipex.txt`.
- Quick start: `./run.sh` (creates venv, installs deps, downloads models, runs WebUI).
- Direct WebUI launch: `python infer-web.py`. Docker: `docker compose up --build` (expects NVIDIA GPU, exposes 7865).
- Meaningful runtime work requires external assets: `ffmpeg`/`ffprobe`, HuBERT, RMVPE, pretrained weights, UVR5 weights, user `.pth` weights, optional FAISS `.index` files. `tools/dlmodels.sh` downloads them.

## Environment and Data Paths

Repo-local `.env` defines runtime roots:

- `weight_root = assets/weights`
- `weight_uvr5_root = assets/uvr5_weights`
- `index_root = logs`
- `outside_index_root = assets/indices`
- `rmvpe_root = assets/rmvpe`

Do not commit model weights, indices, logs, temporary audio, virtualenvs, `runtime/`, downloaded ffmpeg binaries, or generated training outputs. `configs/config.py` copies version configs into `configs/inuse/` on startup and mutates those copies — treat `configs/inuse/` as runtime state.

## Validation

Fast local verification scripts (plain python, no pytest) live in `tests/`:

```bash
.venv/bin/python tests/smoke_imports.py    # imports every refactor-sensitive module
.venv/bin/python tests/model_construct.py  # state_dict fingerprints of all synthesizers
.venv/bin/python tests/webui_build.py      # builds the full Gradio app headlessly
.venv/bin/python tests/pipeline_equiv.py   # silence-search vectorization equivalence
.venv/bin/python tools/realtime/test_protocol.py  # realtime server end-to-end
```

CI also runs a training pipeline smoke test (preprocess → F0 extraction → feature extraction); see `.github/workflows/unitest.yml`.

For small changes, prefer the narrowest useful check: `python -m py_compile <files>` for syntax, `black --check <files>` for formatting. Run the WebUI/API/CLI only when dependencies and model assets exist locally.

Formatting is **Black**, enforced by CI on `main`/`dev`:

```bash
black .
```

## Coding Notes

- Most modules assume the process starts from the repo root and append `os.getcwd()` to `sys.path`.
- Device behavior is centralized in `configs/config.py`: CUDA, MPS, CPU, XPU/IPEX, DirectML, and half-precision selection.
- FAISS index paths are discovered from `logs/` and `assets/indices`; trained indexes are normalized from `trained` to `added` in user-facing paths.
- F0 methods include `pm`, `harvest`, `crepe`, `rmvpe`, and realtime-specific `fcpe` paths; RMVPE is the recommended default.
- Realtime code uses multiprocessing queues for Harvest F0 extraction and `sounddevice` devices; avoid importing or launching it in tests unless audio devices are expected.
- Audio paths may contain Unicode control characters — be careful in file path handling.
- Use a parser for structured edits to files under `configs/` and `i18n/locale/`; avoid string manipulation on JSON.

## Git Hygiene

- Do not revert unrelated user changes.
- Keep generated directories and large binary assets out of commits.
