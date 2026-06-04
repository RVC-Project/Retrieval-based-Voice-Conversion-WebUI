#!/usr/bin/env python3
"""Bootstrap helper for this RVC repo.

Features:
- detects OS and GPU backend
- selects the correct requirements file
- finds/uses supported Python (3.8-3.10)
- creates/upgrades a virtualenv
- pins pip to <24.1 for compatibility
- installs dependencies and fixes gradio-client
- verifies and optionally downloads required model assets
- checks FFmpeg and launches infer-web.py
"""

import platform
import subprocess
import shutil
import sys
import venv
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
VENV_DIR = PROJECT_ROOT / ".venv"

SUPPORTED_PYTHON_MIN = (3, 8)
SUPPORTED_PYTHON_MAX = (3, 11)
SUPPORTED_PIP_MAX = "24.1"

REQUIREMENTS_MAP = {
    "windows": {
        "nvidia": "requirements.txt",
        "amd": "requirements-dml.txt",
        "intel": "requirements-dml.txt",
        "default": "requirements.txt",
    },
    "linux": {
        "nvidia": "requirements.txt",
        "amd": "requirements-amd.txt",
        "intel": "requirements-ipex.txt",
        "default": "requirements.txt",
    },
    "macos": {
        "default": "requirements.txt",
    },
    "unknown": {
        "default": "requirements.txt",
    },
}

ASSETS_REQUIRED = [
    "assets/hubert/hubert_inputs.pth",
    "assets/hubert/hubert_base.pt",
    "assets/rmvpe/rmvpe_inputs.pth",
    "assets/rmvpe/rmvpe.pt",
    "assets/Synthesizer_inputs.pth",
]

OPTIONAL_DOWNLOAD_FILES = {
    "assets/hubert/hubert_base.pt":
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt",
    "assets/rmvpe/rmvpe.pt":
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
    "assets/rmvpe/rmvpe.onnx":
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.onnx",
}

# ============================================================
# LOGGING
# ============================================================

def log(msg):
    print(f"[BOOTSTRAP] {msg}")

# ============================================================
# OS DETECTION
# ============================================================

def detect_os():
    os_name = platform.system().lower()

    if os_name == "windows":
        return "windows"

    if os_name == "linux":
        return "linux"

    if os_name == "darwin":
        return "macos"

    return "unknown"

# ============================================================
# VENV
# ============================================================

def python_version_ok(python_executable):
    try:
        if isinstance(python_executable, (list, tuple)):
            cmd = list(python_executable)
        else:
            cmd = [str(python_executable)]

        cmd += ["-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"]
        output = subprocess.check_output(
            cmd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        version = tuple(int(x) for x in output.split("."))
        return SUPPORTED_PYTHON_MIN <= version < SUPPORTED_PYTHON_MAX, version
    except Exception:
        return False, None


def find_compatible_python():
    current = Path(sys.executable)
    ok, version = python_version_ok(current)
    if ok:
        log(f"Current Python interpreter is supported: {current} -> {version}")
        return current, version

    candidates = []
    if detect_os() == "windows":
        if shutil.which("py"):
            candidates.extend([
                ["py", "-3.10"],
                ["py", "-3.9"],
                ["py", "-3.8"],
            ])
        candidates.extend([
            ["python3.10"],
            ["python3.9"],
            ["python3.8"],
        ])
    else:
        candidates.extend([
            ["python3.10"],
            ["python3.9"],
            ["python3.8"],
            ["python"],
        ])

    for candidate in candidates:
        try:
            if shutil.which(candidate[0]) is None:
                continue
        except Exception:
            pass

        log(f"Checking Python candidate: {candidate}")
        ok, version = python_version_ok(candidate)
        if ok:
            log(f"Found compatible Python candidate: {candidate} -> {version}")
            return candidate, version

    current_ver = tuple(sys.version_info[:3])
    raise RuntimeError(
        f"Unsupported Python version {current_ver}. "
        f"This repository requires Python {SUPPORTED_PYTHON_MIN[0]}.{SUPPORTED_PYTHON_MIN[1]} "
        f"through {SUPPORTED_PYTHON_MAX[0] - 1}.x. "
        "Please install Python 3.10 or 3.9 and run bootstrap with that interpreter. "
        "On Windows, use: py -3.10 bootstrap.py"
    )


def create_venv(python_executable):
    if VENV_DIR.exists():
        existing_python = get_venv_python()
        ok, existing_version = python_version_ok(existing_python)
        if ok:
            log(f"Virtual environment already exists and is compatible ({existing_version}).")
            return

        log(
            "Existing virtual environment is incompatible with this repo. "
            "Removing and recreating with a supported Python interpreter."
        )
        shutil.rmtree(VENV_DIR)

    log(f"Creating virtual environment with {python_executable}...")
    if isinstance(python_executable, (list, tuple)):
        cmd = list(python_executable)
    else:
        cmd = [str(python_executable)]

    cmd += ["-m", "venv", str(VENV_DIR), "--upgrade-deps"]
    subprocess.run(cmd, check=True)


def get_venv_python():
    if detect_os() == "windows":
        return VENV_DIR / "Scripts" / "python.exe"

    return VENV_DIR / "bin" / "python"

# ============================================================
# COMMANDS
# ============================================================

def run(cmd, check=True):
    log(" ".join(str(x) for x in cmd))
    return subprocess.run(cmd, check=check)

def capture(cmd):
    try:
        return subprocess.check_output(
            cmd,
            stderr=subprocess.DEVNULL,
            text=True
        )
    except Exception:
        return ""

# ============================================================
# GPU DETECTION
# ============================================================

def has_nvidia():
    return shutil.which("nvidia-smi") is not None

def detect_linux_gpu():
    try:
        data = capture(["lspci"]).lower()

        if "nvidia" in data:
            return "nvidia"

        if "amd" in data or "radeon" in data:
            return "amd"

        if "intel" in data:
            return "intel"

    except Exception:
        pass

    return None

def detect_windows_gpu():
    try:
        cmd = [
            "wmic",
            "path",
            "win32_VideoController",
            "get",
            "name"
        ]

        output = capture(cmd).lower()

        if "nvidia" in output:
            return "nvidia"

        if "amd" in output or "radeon" in output:
            return "amd"

        if "intel" in output:
            return "intel"

    except Exception:
        pass

    return None

def detect_gpu():
    os_name = detect_os()

    if has_nvidia():
        return "nvidia"

    if os_name == "linux":
        return detect_linux_gpu()

    if os_name == "windows":
        return detect_windows_gpu()

    return None

# ============================================================
# BACKEND SELECTION
# ============================================================

def choose_requirements():
    os_name = detect_os()
    gpu = detect_gpu()
    backend_map = REQUIREMENTS_MAP.get(os_name, REQUIREMENTS_MAP["unknown"])
    req = backend_map.get(gpu) or backend_map.get("default")
    if not req:
        raise RuntimeError(f"No requirements mapping for os={os_name} gpu={gpu}")
    return req, os_name, gpu

# ============================================================
# REQUIREMENTS
# ============================================================

def get_pip_version(python):
    try:
        output = subprocess.check_output(
            [str(python), "-m", "pip", "--version"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return output
    except Exception:
        return None


def install_requirements(requirements_file):
    req_path = PROJECT_ROOT / requirements_file

    if not req_path.exists():
        raise RuntimeError(f"Missing requirements file: {requirements_file}")

    python = str(get_venv_python())
    log(f"Installing requirements with {python}: {requirements_file}")
    log(f"Initial pip version: {get_pip_version(python) or 'unknown'}")

    run([
        python,
        "-m",
        "pip",
        "install",
        f"pip<{SUPPORTED_PIP_MAX}"
    ])

    log(f"Pinned pip to <{SUPPORTED_PIP_MAX}: {get_pip_version(python) or 'unknown'}")

    run([
        python,
        "-m",
        "pip",
        "install",
        "-r",
        str(req_path)
    ])

    log("Ensuring Gradio client compatibility...")
    run([
        python,
        "-m",
        "pip",
        "install",
        "gradio-client==0.2.7"
    ])

# ============================================================
# ASSET VERIFICATION
# ============================================================

def check_assets():
    missing = [path for path in ASSETS_REQUIRED if not (PROJECT_ROOT / path).exists()]
    if missing:
        raise RuntimeError(
            "Missing required assets:\n" + "\n".join(f"- {p}" for p in missing)
        )
    log(f"All required assets present ({len(ASSETS_REQUIRED)} items).")


def download_file(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    log(f"Downloading {destination.name} from {url}")
    import urllib.request
    urllib.request.urlretrieve(url, destination)


def download_optional_models():
    log("Checking optional model files...")
    for path, url in OPTIONAL_DOWNLOAD_FILES.items():
        target = PROJECT_ROOT / path
        if target.exists():
            log(f"Optional model already exists: {path}")
            continue
        download_file(url, target)
    log("Optional model check complete.")

# ============================================================
# FFMPEG
# ============================================================

def ffmpeg_exists():
    if shutil.which("ffmpeg"):
        return True
    if detect_os() == "windows":
        return (PROJECT_ROOT / "ffmpeg.exe").exists() and (PROJECT_ROOT / "ffprobe.exe").exists()
    return False


def check_ffmpeg():
    if ffmpeg_exists():
        log("FFmpeg found.")
        return

    os_name = detect_os()

    log("FFmpeg not detected.")

    if os_name == "windows":
        log(
            "Please place ffmpeg.exe and ffprobe.exe "
            "in the project root."
        )

    elif os_name == "linux":
        log(
            "Install FFmpeg:\n"
            "Ubuntu/Debian: sudo apt install ffmpeg\n"
            "Arch: sudo pacman -S ffmpeg\n"
            "Fedora: sudo dnf install ffmpeg"
        )

    elif os_name == "macos":
        log(
            "Install FFmpeg:\n"
            "brew install ffmpeg"
        )

# ============================================================
# VERIFY
# ============================================================

def verify_torch():
    python = str(get_venv_python())

    code = r"""
import torch
print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
"""

    run([
        python,
        "-c",
        code
    ])

# ============================================================
# LAUNCH
# ============================================================

def launch():
    python = str(get_venv_python())

    target = PROJECT_ROOT / "infer-web.py"

    if not target.exists():
        raise RuntimeError(
            "infer-web.py not found."
        )

    log("Launching RVC...")

    subprocess.Popen([
        python,
        str(target)
    ])

# ============================================================
# MAIN
# ============================================================

def main():
    log("Starting bootstrap...")

    python_executable, version = find_compatible_python()
    if isinstance(python_executable, (list, tuple)):
        python_label = " ".join(python_executable)
    else:
        python_label = str(python_executable)

    log(f"Using Python interpreter: {python_label} ({version[0]}.{version[1]}.{version[2]})")
    create_venv(python_executable)

    requirements_file, os_name, gpu = choose_requirements()
    log(f"Detected OS: {os_name}")
    log(f"Detected GPU: {gpu}")
    log(f"Selected requirements: {requirements_file}")

    install_requirements(requirements_file)
    download_optional_models()
    check_assets()

    check_ffmpeg()
    verify_torch()
    launch()

if __name__ == "__main__":
    main()
