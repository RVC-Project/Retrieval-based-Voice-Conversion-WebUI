import os
from pathlib import Path
import hashlib
import requests
from io import BytesIO
import logging

logger = logging.getLogger(__name__)


def sha256(f) -> str:
    sha256_hash = hashlib.sha256()
    # Read and update hash in chunks of 4M
    for byte_block in iter(lambda: f.read(4 * 1024 * 1024), b""):
        sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def check_model(dir_name: Path, model_name: str, hash: str) -> bool:
    target = dir_name / model_name
    relname = str(target)
    relname = relname[relname.rindex("assets/") :]
    logger.debug(f"checking {relname}...")
    if not os.path.exists(target):
        logger.info(f"{target} not exist.")
        return False
    with open(target, "rb") as f:
        digest = sha256(f)
        if digest != hash:
            logger.info(f"{target} sha256 hash mismatch.")
            logger.info(f"expected: {hash}")
            logger.info(f"real val: {digest}")
            os.remove(str(target))
            return False
    return True


def check_all_assets() -> bool:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    logger.info("checking hubret & rmvpe...")

    if not check_model(
        BASE_DIR / "assets/hubert",
        "hubert_base.pt",
        os.environ["sha256_hubert_base_pt"],
    ):
        return False
    if not check_model(
        BASE_DIR / "assets/rmvpe", "rmvpe.pt", os.environ["sha256_rmvpe_pt"]
    ):
        return False
    if not check_model(
        BASE_DIR / "assets/rmvpe", "rmvpe.onnx", os.environ["sha256_rmvpe_onnx"]
    ):
        return False

    rvc_models_dir = BASE_DIR / "assets/pretrained"
    logger.info("checking pretrained models...")
    model_names = [
        "D32k.pth",
        "D40k.pth",
        "D48k.pth",
        "G32k.pth",
        "G40k.pth",
        "G48k.pth",
        "f0D32k.pth",
        "f0D40k.pth",
        "f0D48k.pth",
        "f0G32k.pth",
        "f0G40k.pth",
        "f0G48k.pth",
    ]
    for model in model_names:
        menv = model.replace(".", "_")
        if not check_model(rvc_models_dir, model, os.environ[f"sha256_v1_{menv}"]):
            return False

    rvc_models_dir = BASE_DIR / "assets/pretrained_v2"
    logger.info("checking pretrained models v2...")
    for model in model_names:
        menv = model.replace(".", "_")
        if not check_model(rvc_models_dir, model, os.environ[f"sha256_v2_{menv}"]):
            return False

    logger.info("checking uvr5_weights...")
    rvc_models_dir = BASE_DIR / "assets/uvr5_weights"
    model_names = [
        "HP2-人声vocals+非人声instrumentals.pth",
        "HP2_all_vocals.pth",
        "HP3_all_vocals.pth",
        "HP5-主旋律人声vocals+其他instrumentals.pth",
        "HP5_only_main_vocal.pth",
        "VR-DeEchoAggressive.pth",
        "VR-DeEchoDeReverb.pth",
        "VR-DeEchoNormal.pth",
    ]
    for model in model_names:
        menv = model.replace(".", "_")
        if not check_model(rvc_models_dir, model, os.environ[f"sha256_uvr5_{menv}"]):
            return False
    if not check_model(
        BASE_DIR / "assets/uvr5_weights/onnx_dereverb_By_FoxJoy",
        "vocals.onnx",
        os.environ[f"sha256_uvr5_vocals_onnx"],
    ):
        return False

    logger.info("all assets are already latest.")
    return True


def download_and_extract_tar_gz(url: str, folder: str):
    import tarfile

    logger.info(f"downloading {url}")
    response = requests.get(url, stream=True, timeout=(5, 10))
    with BytesIO() as out_file:
        out_file.write(response.content)
        out_file.seek(0)
        logger.info(f"downloaded.")
        with tarfile.open(fileobj=out_file, mode="r:gz") as tar:
            tar.extractall(folder)
        logger.info(f"extracted into {folder}")


def download_and_extract_zip(url: str, folder: str):
    import zipfile

    logger.info(f"downloading {url}")
    response = requests.get(url, stream=True, timeout=(5, 10))
    with BytesIO() as out_file:
        out_file.write(response.content)
        out_file.seek(0)
        logger.info(f"downloaded.")
        with zipfile.ZipFile(out_file) as zip_ref:
            zip_ref.extractall(folder)
        logger.info(f"extracted into {folder}")


def download_dns_yaml(url: str, folder: str):
    logger.info(f"downloading {url}")
    response = requests.get(url, stream=True, timeout=(5, 10))
    with open(f"{folder}/dns.yaml", "wb") as out_file:
        out_file.write(response.content)
        logger.info(f"downloaded into {folder}")


def download_all_assets(tmpdir: str, version="0.2.2"):
    import subprocess
    import platform

    archs = {
        "aarch64": "arm64",
        "armv8l": "arm64",
        "arm64": "arm64",
        "x86": "386",
        "i386": "386",
        "i686": "386",
        "386": "386",
        "x86_64": "amd64",
        "x64": "amd64",
        "amd64": "amd64",
    }
    system_type = platform.system().lower()
    architecture = platform.machine().lower()
    is_win = architecture == "windows"

    architecture = archs.get(architecture, None)
    if not architecture:
        logger.error(f"architecture {architecture} is not supported")
        exit(1)
    try:
        BASE_URL = (
            "https://github.com/RVC-Project/RVC-Models-Downloader/releases/download/"
        )
        suffix = "zip" if is_win else "tar.gz"
        RVCMD_URL = BASE_URL + f"v{version}/rvcmd_{system_type}_{architecture}.{suffix}"
        cmdfile = tmpdir + "/rvcmd"
        if is_win:
            download_and_extract_zip(RVCMD_URL, tmpdir)
            cmdfile += ".exe"
        else:
            download_and_extract_tar_gz(RVCMD_URL, tmpdir)
            os.chmod(cmdfile, 0o755)
        subprocess.run([cmdfile, "-notui", "-w", "0", "assets/all"])
    except Exception:
        BASE_URL = "https://raw.gitcode.com/u011570312/RVC-Models-Downloader/assets/"
        suffix = {
            "darwin_amd64": "421",
            "darwin_arm64": "422",
            "linux_386": "423",
            "linux_amd64": "424",
            "linux_arm64": "425",
            "windows_386": "426",
            "windows_amd64": "427",
        }[f"{system_type}_{architecture}"]
        RVCMD_URL = BASE_URL + suffix
        download_dns_yaml(
            "https://raw.gitcode.com/u011570312/RVC-Models-Downloader/raw/main/dns.yaml",
            tmpdir,
        )
        if is_win:
            download_and_extract_zip(RVCMD_URL, tmpdir)
            cmdfile += ".exe"
        else:
            download_and_extract_tar_gz(RVCMD_URL, tmpdir)
            os.chmod(cmdfile, 0o755)
        subprocess.run(
            [cmdfile, "-notui", "-w", "0", "-dns", f"{tmpdir}/dns.yaml", "assets/all"]
        )
