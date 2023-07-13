#!/bin/bash

if [[ "$(uname)" == "Darwin" ]]; then
  # macOS specific env:
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
elif [[ "$(uname)" != "Linux" ]]; then
  echo "Unsupported operating system."
  exit 1
fi

requirements_file="requirements.txt"

# Check if Python 3.8 is installed
if ! command -v python3.8 &> /dev/null; then
  echo "Python 3.8 not found. Attempting to install..."
  if [[ "$(uname)" == "Darwin" ]] && command -v brew &> /dev/null; then
    brew install python@3.8
  elif [[ "$(uname)" == "Linux" ]] && command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install python3.8
  else
    echo "Please install Python 3.8 manually."
    exit 1
  fi
fi

# Check if required packages are installed and install them if not
if [ -f "${requirements_file}" ]; then
  installed_packages=$(python3.8 -m pip freeze)
  while IFS= read -r package; do
    [[ "${package}" =~ ^#.* ]] && continue
    package_name=$(echo "${package}" | sed 's/[<>=!].*//')
    if ! echo "${installed_packages}" | grep -q "${package_name}"; then
      echo "${package_name} not found. Attempting to install..."
      python3.8 -m pip install --upgrade "${package}"
    fi
  done < "${requirements_file}"
else
  echo "${requirements_file} not found. Please ensure the requirements file with required packages exists."
  exit 1
fi

# Run the main script
python3.8 infer-web.py --pycmd python3.8