#!/bin/bash
set -e

# Entrypoint Script for RVC WebUI with ROCm 5.4.2 for RX580 / Polaris / gfx803 AMD GPU
# Created by Levent Sunay (lsunay) - May-April 2025
#
# Description:
# This script serves as the entrypoint for the Retrieval-based-Voice-Conversion-WebUI (RVC) Docker container.
# It activates the Python virtual environment, downloads necessary pre-trained models if not already present,
# and starts the RVC server on the specified port.

APP_DIR="/app" # Main directory where RVC files are located
echo "--- RVC Entrypoint Started ---"
cd "$APP_DIR"
echo "Current directory: $(pwd)"
echo "Python3 version: $(python3 --version)"

VENV_PATH="./.venv"
echo "Entrypoint: Activating Python venv for RVC at ${VENV_PATH}..."
if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
    echo "RVC Python venv activated. Active Python: $(python --version)"
else
    echo "ERROR: RVC Python venv not found at ${VENV_PATH}/bin/activate"
    exit 1
fi

# === Model Download (Only if required files are not present under assets) ===
ASSETS_DIR="./assets"
HUBERT_FILE="${ASSETS_DIR}/hubert/hubert_base.pt"
if [ ! -f "${HUBERT_FILE}" ]; then
    echo "Required model files not found in ${ASSETS_DIR}. Downloading RVC pre-models..."
    mkdir -p "${ASSETS_DIR}/pretrained_v2" "${ASSETS_DIR}/uvr5_weights" "${ASSETS_DIR}/hubert" "${ASSETS_DIR}/rmvpe"
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/D40k.pth -d "${ASSETS_DIR}/pretrained_v2/" -o D40k.pth
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/G40k.pth -d "${ASSETS_DIR}/pretrained_v2/" -o G40k.pth
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D40k.pth -d "${ASSETS_DIR}/pretrained_v2/" -o f0D40k.pth
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G40k.pth -d "${ASSETS_DIR}/pretrained_v2/" -o f0G40k.pth
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP2-人声vocals+非人声instrumentals.pth -d "${ASSETS_DIR}/uvr5_weights/" -o HP2-人声vocals+非人声instrumentals.pth
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP5-主旋律人声vocals+其他instrumentals.pth -d "${ASSETS_DIR}/uvr5_weights/" -o HP5-主旋律人声vocals+其他instrumentals.pth
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt -d "${ASSETS_DIR}/hubert/" -o hubert_base.pt
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt -d "${ASSETS_DIR}/rmvpe/" -o rmvpe.pt
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.onnx -d "${ASSETS_DIR}/rmvpe/" -o rmvpe.onnx
    echo "RVC pre-models downloaded to ${ASSETS_DIR}."
else
    echo "RVC pre-models already found in ${ASSETS_DIR}."
fi
# === End of Model Download ===

echo "Starting RVC server..."
if [ -f "infer-web.py" ]; then
    echo "Found infer-web.py, executing with port ${RVC_PORT:-7865}"
    # Removed --host 0.0.0.0 argument.
    # --pycmd python (or python3) can be added based on RVC's expectation.
    # Gradio typically uses server_name="0.0.0.0" internally in the code.
    exec python infer-web.py --port ${RVC_PORT:-7865} --pycmd python "$@" # <-- Fixed line
elif [ -f "run.sh" ]; then # If RVC's own run.sh exists, it can be used instead.
    echo "Found run.sh, executing it..."
    # Check if run.sh accepts arguments.
    exec ./run.sh # It might accept arguments like --port ${RVC_PORT:-7865}
else
    echo "ERROR: Main RVC script (infer-web.py or run.sh) not found in $(pwd)"
    exit 1
fi
