#!/bin/sh

if [ "$(uname)" = "Darwin" ]; then
  # macOS specific env:
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
elif [ "$(uname)" != "Linux" ]; then
  echo "Unsupported operating system."
  exit 1
fi

find_python() {
  if [ -n "${PYTHON:-}" ] && command -v "${PYTHON}" >/dev/null 2>&1; then
    printf "%s\n" "${PYTHON}"
    return 0
  fi

  for candidate in python3.9 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      printf "%s\n" "${candidate}"
      return 0
    fi
  done

  return 1
}

PYTHON_CMD="$(find_python || true)"
if [ -z "${PYTHON_CMD}" ]; then
  echo "Python 3.9 is required."
  echo "Set PYTHON=/path/to/python before running this script if it is not on PATH."
  exit 1
fi

if [ -d ".venv" ]; then
  echo "Activate venv..."
  . .venv/bin/activate
else
  echo "Create venv..."
  requirements_file="requirements.txt"

  "${PYTHON_CMD}" -m venv .venv
  . .venv/bin/activate

  if [ -f "${requirements_file}" ]; then
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install "torch==2.8.0" "torchaudio==2.8.0" "torchvision==0.23.0"
    python -m pip install -r "${requirements_file}"
  else
    echo "${requirements_file} not found. Please ensure the requirements file with required packages exists."
    exit 1
  fi
fi

# Run the real-time voice conversion
echo "Starting real-time voice conversion..."
python tools/rvc_for_realtime.py
