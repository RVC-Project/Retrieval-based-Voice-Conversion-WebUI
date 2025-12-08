# RVC WebUI Setup Documentation

## Environment

- macOS (Apple Silicon)
- Python 3.10 (Conda)
- PyTorch with MPS support

## Problems and Solutions

### 1. fairseq Installation Failure

**Error**: `omegaconf` metadata parsing error with pip 25.x

**Fix**:
```bash
pip install "pip<24.1"
pip install fairseq==0.12.2
```

### 2. Gradio Version Mismatch

**Error**: `concurrency_count` parameter not recognized (wrong gradio version installed)

**Fix**:
```bash
pip install gradio==3.34.0
```

### 3. gradio_client.serializing Module Not Found

**Error**: `ModuleNotFoundError: No module named 'gradio_client.serializing'`

**Cause**: Modern gradio-client (2.x) removed the serializing module that gradio 3.34.0 expects

**Fix**:
```bash
pip install gradio-client==0.2.7
```

## Working Installation Sequence

```bash
conda create -n rvc python=3.10 -y
conda activate rvc

pip install "pip<24.1"
pip install torch torchvision torchaudio
pip install -r requirements.txt
pip install fairseq==0.12.2
pip install gradio==3.34.0
pip install gradio-client==0.2.7

python tools/download_models.py
python infer-web.py
```

## Verification

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
python -c "from gradio_client.serializing import Serializable; print('OK')"
```

## Access

WebUI runs at: http://localhost:7865
