# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RVC (Retrieval-based Voice Conversion) is a voice conversion framework based on VITS that enables real-time voice synthesis and conversion. The project includes:
- WebUI for training and inference (Gradio-based)
- Real-time voice conversion with low latency (~90-170ms)
- Voice separation (UVR5)
- Voice pitch extraction (RMVPE, Crepe)
- Multiple hardware acceleration backends (CUDA, DirectML, ROCM, IPEX)
- ONNX export and TensorRT support

## Python Environment

- **Supported versions**: Python 3.10-3.13 (DirectML limited to 3.10-3.12, IPEX limited to 3.10)
- **Setup commands**:
  - Quick start with shell script: `./run.sh` (Linux/macOS) - creates venv, installs deps, downloads models, runs WebUI
  - Manual venv creation: `./venv.sh`
  - Dependency installation: `pip install -r requirements.txt` (NVIDIA GPU default)
  - Alternative hardware backends: `requirements-dml.txt` (DirectML), `requirements-amd.txt` (ROCM), `requirements-ipex.txt` (Intel IPEX)
  - Poetry-based setup: `poetry env use /path/to/python && poetry lock && poetry install`

## Core Entry Points

1. **Web UI** (`infer-web.py`): thin launcher for the Gradio interface; the actual UI lives in the `web/` package:
   - `web/runtime.py`: one-time setup (env, Config/VC/i18n singletons, GPU detection, model+index discovery)
   - `web/train_ops.py`: training subprocess orchestration and Gradio callbacks
   - `web/tabs/{inference,uvr5,train,ckpt,onnx,faq}.py`: one module per tab, each exposing `build()`
   - `web/app.py`: `build_app()` assembles the Blocks (used headlessly by `tests/webui_build.py`)
2. **GUI** (`gui_v1.py`): FreeSimpleGUI-based desktop interface
3. **REST APIs**:
   - `api_240604.py`: FastAPI-based API (latest version)
   - `api_231006.py`: Legacy API endpoint
4. **CLI Tools** (`tools/`):
   - `infer_cli.py`: Command-line inference
   - `rvc_for_realtime.py`: Real-time voice conversion
   - `infer_batch_rvc.py`: Batch processing
   - `export_onnx.py`: ONNX model export
   - `calc_rvc_model_similarity.py`: Model similarity calculation

## Architecture

### Key Modules

**Voice Conversion Pipeline** (`infer/modules/vc/`):
- `modules.py`: Core VC class handling model loading and inference
- `pipeline.py`: Inference pipeline with pitch detection and voice conversion
- `utils.py`: Audio processing utilities

**Training** (`infer/modules/train/`):
- `train.py`: Distributed training script
- `preprocess.py`: Audio preprocessing (resampling, loudness normalization)
- `extract/`: Feature extraction modules (F0 extraction, HuBERT features)

**Voice Separation** (`infer/modules/uvr5/`):
- UVR5-based voice/instrumental separation

**Model Definitions** (`infer/lib/infer_pack/`):
- VITS-based synthesizer models in different configurations (v1: 256k/40k/48k, v2: 48k/32k)
- Models support both with and without F0 (pitch) information
- Split across `encoders.py` (TextEncoder, PosteriorEncoder, ResidualCouplingBlock), `generators.py` (Generator, GeneratorNSF, SineGen, SourceModuleHnNSF), `discriminators.py`, and `models.py` (the four `SynthesizerTrn*` classes); `models.py` re-exports everything, so always import from `infer.lib.infer_pack.models`
- **Never rename the `SynthesizerTrnMs{256,768}NSFsid[_nono]` classes** — checkpoint loading instantiates them by these exact names in `infer/modules/vc/modules.py`, `infer/lib/jit/get_synthesizer.py`, and `infer/modules/train/train.py`
- `models_onnx.py` / `attentions_onnx.py` are intentionally separate ONNX-traceable variants; do not merge them with the torch versions

**Realtime stack** (`tools/realtime/` + `gui_v1.py`):
- Shared helpers: `tools/realtime/dsp.py` (phase_vocoder, printt), `tools/realtime/devices.py` (sounddevice queries), `tools/realtime/harvest_worker.py` (Harvest F0 process) — used by both `gui_v1.py` and `tools/realtime/engine.py`
- `infer/lib/rtrvc.py` is the canonical realtime RVC engine (used by gui_v1 and the Electron server); `tools/rvc_for_realtime.py` is a simplified variant kept for `api_240604.py`/`api_231006.py`

**Hardware Acceleration**:
- `infer/modules/ipex/`: Intel IPEX optimization (attention, gradients, hijacks)
- CUDA/DirectML/ROCM integration via PyTorch

### Configuration

- `configs/config.py`: Device detection, model paths, performance settings
- Supports singleton pattern to cache config across app
- Auto-detects GPU capability and selects appropriate model precision

### Audio Processing

- `infer/lib/audio/`: Audio I/O, feature extraction using librosa, scipy
- F0 predictors: RMVPE (default), Crepe, Harvest, PM

## Development Tasks

### Code Formatting

The project uses **Black** for code formatting:
```bash
black .
```
This is enforced via CI/CD on push to `main`/`dev` branches. PRs will be auto-formatted if needed.

### Running Tests

Fast local verification scripts (plain python, no pytest) live in `tests/`:
```bash
.venv/bin/python tests/smoke_imports.py    # imports every refactor-sensitive module
.venv/bin/python tests/model_construct.py  # state_dict fingerprints of all synthesizers
.venv/bin/python tests/webui_build.py      # builds the full Gradio app headlessly
.venv/bin/python tests/pipeline_equiv.py   # silence-search vectorization equivalence
.venv/bin/python tools/realtime/test_protocol.py  # realtime server end-to-end
```

Unit tests are in the CI/CD pipeline (`.github/workflows/unitest.yml`). To run locally:
```bash
# Install dependencies with ffmpeg, aria2
sudo apt update && sudo apt install ffmpeg aria2
python -m pip install torch torchvision torchaudio -r requirements.txt

# Download test model (HuBERT)
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
  https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt \
  -d ./ -o hubert_base.pt

# Test preprocessing and feature extraction
mkdir -p logs/mi-test
touch logs/mi-test/preprocess.log
python infer/modules/train/preprocess.py logs/mute/0_gt_wavs 48000 8 logs/mi-test True 3.7

touch logs/mi-test/extract_f0_feature.log
python infer/modules/train/extract/extract_f0_print.py logs/mi-test $(nproc) pm
python infer/modules/train/extract_feature_print.py cpu 1 0 0 logs/mi-test v1 True
```

Test coverage focuses on the training pipeline: preprocessing, F0 extraction, and feature extraction.

### Running the Application

- **WebUI**: `python infer-web.py [--pycmd python]`
- **Real-time VC**: `python tools/rvc_for_realtime.py` (requires audio input/output)
- **Batch inference**: `python tools/infer_batch_rvc.py [options]`
- **API server**: `python api_240604.py --listen 0.0.0.0 --port 8000`

### Key Dependencies

- **Audio**: librosa, soundfile, pydub, scipy, pyworld
- **Model**: torch, fairseq (custom fork), faiss-cpu
- **Feature extraction**: HuBERT (from fairseq), torchcrepe, torchfcpe (F0)
- **Web**: gradio, fastapi, uvicorn
- **Utilities**: numpy, scikit-learn, matplotlib, tqdm

### Model Weights

Models are downloaded automatically by `tools/dlmodels.sh` on first run. Common model locations:
- `assets/weights/`: Voice model checkpoints (.pth files)
- `assets/hubert`: HuBERT base model
- `assets/rmvpe`: RMVPE pitch extraction model
- `assets/uvr5_weights/`: Voice separation models
- `assets/pretrained_v2/`: Pretrained v2 generator/discriminator

## Key Design Patterns

### Device Abstraction

The `Config` class abstracts device selection (CUDA, CPU, DirectML, IPEX):
- Uses `torch.cuda.is_available()` to detect GPUs
- Supports half precision (float16) when hardware capable
- Can be overridden via CLI args

### Model Caching

The `VC` class caches loaded models in memory to avoid reloading:
- Loads checkpoint once, reuses for multiple inferences
- Clears cache when switching models or on explicit unload
- Uses `torch.cuda.empty_cache()` to manage GPU memory

### Feature Extraction Pipeline

Two-stage inference:
1. **Preprocessing**: Audio resampling, normalization, chunking
2. **Feature extraction**: HuBERT embeddings, pitch extraction
3. **Conversion**: Synthesizer processes features to produce voice

## Important Notes for Development

- Audio paths may contain Unicode control characters - be aware during file path handling
- The project supports training and inference on consumer-grade GPUs (RTX 30xx+)
- Pitch extraction accuracy is critical - RMVPE is the recommended algorithm
- Training uses distributed training support (DDP) for multi-GPU scenarios
- Poetry support is present but requirements.txt is the primary dependency source
- The codebase has multiple language documentation (zh, en, ja, kr, fr, pt, tr)

## Docker

Docker builds are available with CUDA 11.6 support:
```bash
docker build -t rvc:latest .
docker-compose up  # if docker-compose.yml exists
```

Models are pre-downloaded in the image, ready for inference.

## Recent Changes

Recent work has focused on:
- ONNX export improvements and TensorRT support (#2384)
- NSF inference optimization (#2387)
- Adding warnings for ONNX export edge cases (#2385)
- Removing Unicode control characters from audio file paths (#2334)
