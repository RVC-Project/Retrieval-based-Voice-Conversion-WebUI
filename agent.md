# Agent Notes

## Project Overview

This repository is `Retrieval-based-Voice-Conversion-WebUI`, a Python voice-conversion application built around RVC/VITS, HuBERT feature extraction, FAISS retrieval indices, RMVPE/other F0 extractors, Gradio WebUI, realtime voice conversion, UVR5 vocal separation, and FastAPI/realtime API helpers.

Keep changes small and targeted. `CONTRIBUTING.md` says algorithm changes are generally rejected unless they fix code-level bugs or warnings; UI, translations, and operational fixes are safer contribution areas.

## Main Entry Points

- `infer-web.py`: primary Gradio WebUI for inference, training orchestration, UVR5, model utilities, and checkpoint operations. Default port is `7865`.
- `tools/infer_cli.py`: command-line single-file conversion wrapper around `configs.config.Config` and `infer.modules.vc.modules.VC`.
- `gui_v1.py`: realtime desktop GUI using `FreeSimpleGUI`, `sounddevice`, and realtime RVC helpers.
- `api_240604.py` and `api_231006.py`: FastAPI/realtime API implementations.
- `infer/modules/vc/modules.py`: high-level `VC` class for model loading, inference, batch inference, and index selection.
- `infer/modules/vc/pipeline.py`: core conversion pipeline, F0 extraction methods, RMS mixing, feature/index use.
- `infer/modules/train/`: preprocessing, F0/feature extraction, and training scripts.
- `infer/modules/uvr5/` and `infer/lib/uvr5_pack/`: UVR5 separation code.
- `tools/download_models.py` and `tools/dlmodels.sh`: pretrained asset download helpers.

## Setup And Running

- Python support in CI covers `3.8`, `3.9`, and `3.10`. `pyproject.toml` declares Python `^3.9`, while `run.sh` creates/uses a Python 3.8 virtualenv.
- Install PyTorch first for the target hardware, then install the matching requirements file:
  - NVIDIA/general: `pip install -r requirements.txt`
  - DirectML: `pip install -r requirements-dml.txt`
  - AMD ROCm: `pip install -r requirements-amd.txt`
  - Intel IPEX: `pip install -r requirements-ipex.txt`
- macOS/Linux convenience launch: `sh ./run.sh`
- Direct WebUI launch: `python infer-web.py`
- Docker launch: `docker compose up --build` exposes `7865` and expects NVIDIA GPU support.

The app requires external binaries/assets for meaningful runtime work: `ffmpeg`, `ffprobe`, HuBERT, RMVPE, pretrained RVC weights, UVR5 weights, user `.pth` weights, and optional FAISS `.index` files.

## Environment And Data Paths

Repo-local `.env` defines the important runtime roots:

- `weight_root = assets/weights`
- `weight_uvr5_root = assets/uvr5_weights`
- `index_root = logs`
- `outside_index_root = assets/indices`
- `rmvpe_root = assets/rmvpe`

Generated or large runtime artifacts are intentionally ignored by git. Avoid committing model weights, indices, logs, temporary audio, virtualenvs, `runtime/`, downloaded ffmpeg binaries, or generated training outputs.

`configs/config.py` copies version configs into `configs/inuse/` on startup and mutates those active copies based on device/precision. Treat `configs/inuse/` as runtime state unless the user explicitly asks to inspect or preserve it.

## Validation

There is no dedicated pytest suite in this repo. Existing CI does:

- Format check/fix with `black .`
- A training pipeline smoke test:

```bash
mkdir -p logs/mi-test
touch logs/mi-test/preprocess.log
python infer/modules/train/preprocess.py logs/mute/0_gt_wavs 48000 8 logs/mi-test True 3.7
touch logs/mi-test/extract_f0_feature.log
python infer/modules/train/extract/extract_f0_print.py logs/mi-test $(nproc) pm
python infer/modules/train/extract_feature_print.py cpu 1 0 0 logs/mi-test v1 True
```

For local agent work, prefer the narrowest useful check:

- `python -m py_compile <changed files>` for syntax-level Python changes.
- `black --check <changed files>` if Black is available.
- Run WebUI/API/CLI only when dependencies and model assets exist locally.

## Coding Notes

- Most modules assume the process starts from the repo root and append `os.getcwd()` to `sys.path`.
- Device behavior is centralized in `configs/config.py`: CUDA, MPS fallback, CPU fallback, XPU/IPEX, DirectML, and half precision.
- `infer.modules.vc.modules.VC.get_vc()` loads model checkpoints from `weight_root`; `vc_single()` does inference via `Pipeline`.
- FAISS index paths are discovered from `logs` and `assets/indices`; trained indexes are normalized from `trained` to `added` in user-facing paths.
- F0 methods include `pm`, `harvest`, `crepe`, `rmvpe`, and realtime-specific `fcpe` paths.
- Realtime code uses multiprocessing queues for Harvest F0 extraction and `sounddevice` devices; avoid importing or launching it in tests unless audio devices are expected.
- Many files contain Chinese comments/UI strings. Preserve existing i18n keys and locale JSON structure when editing UI text.
- Use structured JSON edits for files under `configs/` and `i18n/locale/`; avoid hand-built string manipulation where a parser is practical.

## Git Hygiene

- Worktree was clean when this file was created.
- Do not revert unrelated user changes.
- Keep generated directories and large binary assets out of commits.
