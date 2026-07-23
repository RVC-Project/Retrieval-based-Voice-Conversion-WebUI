import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from tqdm import tqdm

from .model_registry import (
    auxiliary_paths_for,
    config_path_for,
    get_model_entry,
    model_path_for,
)


HF_REPO = "baicai1145/pymss"
MS_REPO = "baicai1145/pymss"
HF_BASE_URL = f"https://huggingface.co/{HF_REPO}/resolve/main"
MS_BASE_URL = f"https://www.modelscope.cn/models/{MS_REPO}/resolve/master"
MS_FILES_API = f"https://www.modelscope.cn/api/v1/models/{MS_REPO}/repo/files?Revision=master&Recursive=true"
MODEL_FILE_SUFFIXES = {".ckpt", ".th", ".pth", ".chpt", ".safetensors", ".pt", ".yaml", ".yml", ".json"}
ARIA2C_PATH = shutil.which("aria2c")


class DownloadError(RuntimeError):
    """Base exception raised when model download fails."""

    pass


class DownloadValidationError(DownloadError):
    """Exception raised when a downloaded file fails size or hash validation."""

    pass


def _quote_path(path):
    """Implement the quote path helper.

    Args:
        path (str | os.PathLike): File system path.

    Returns:
        Any: Computed result."""
    return urllib.parse.quote(path, safe="/")


def remote_url(relpath, source="modelscope", endpoint=None):
    """Build the remote download URL for a catalog-relative file path.

    Args:
        relpath (str): Relpath value.
        source (str, optional): Download source name. Defaults to "modelscope".
        endpoint (str | None, optional): Optional custom download endpoint. Defaults to None.

    Returns:
        str: Absolute URL for the requested source."""
    if endpoint:
        return f"{endpoint.rstrip('/')}/{_quote_path(relpath)}"
    if source == "huggingface":
        return f"{HF_BASE_URL}/{_quote_path(relpath)}"
    if source == "hf-mirror":
        return f"https://hf-mirror.com/{HF_REPO}/resolve/main/{_quote_path(relpath)}"
    if source == "modelscope":
        return f"{MS_BASE_URL}/{_quote_path(relpath)}"
    raise ValueError("source must be one of: modelscope, huggingface, hf-mirror")


def _read_json_url(url, timeout=30):
    """Read json url.

    Args:
        url (str): Url value.
        timeout (int, optional): Network timeout in seconds. Defaults to 30.

    Returns:
        Any: Computed result."""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def fetch_modelscope_file_index(timeout=30):
    """Fetch modelscope file index.

    Args:
        timeout (int, optional): Network timeout in seconds. Defaults to 30.

    Returns:
        Any: Computed result."""
    data = _read_json_url(MS_FILES_API, timeout=timeout)
    files = data.get("Data", {}).get("Files", [])
    return {item["Path"]: item for item in files if item.get("Type") == "blob"}


def _sha256(path):
    """Implement the sha256 helper.

    Args:
        path (str | os.PathLike): File system path.

    Returns:
        Any: Computed result."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _expected_size_and_hash(relpath, source_index):
    """Implement the expected size and hash helper.

    Args:
        relpath (str): Relpath value.
        source_index (Any): Source index value.

    Returns:
        Any: Computed result."""
    if not source_index:
        return None, ""
    item = source_index.get(relpath, {})
    size = item.get("Size")
    sha256 = item.get("Sha256") or ""
    return int(size) if size else None, sha256


def _already_valid(path, expected_size=None, expected_sha256=""):
    """Implement the already valid helper.

    Args:
        path (str | os.PathLike): File system path.
        expected_size (Any, optional): Expected size value. Defaults to None.
        expected_sha256 (Any, optional): Expected sha256 value. Defaults to ''.

    Returns:
        Any: Computed result."""
    if not path.is_file():
        return False
    if expected_size is not None and path.stat().st_size != expected_size:
        return False
    if expected_sha256 and _sha256(path) != expected_sha256:
        return False
    return True


def _cleanup_partial_download(tmp):
    """Implement the cleanup partial download helper.

    Args:
        tmp (Any): Tmp value.

    Returns:
        None: This callable completes for its side effects."""
    for path in (tmp, Path(str(tmp) + ".aria2")):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _validate_downloaded_file(path, dest, expected_size=None, expected_sha256=""):
    """Validate downloaded file.

    Args:
        path (str | os.PathLike): File system path.
        dest (Any): Dest value.
        expected_size (Any, optional): Expected size value. Defaults to None.
        expected_sha256 (Any, optional): Expected sha256 value. Defaults to ''.

    Returns:
        None: This callable completes for its side effects."""
    if expected_size is not None and path.stat().st_size != expected_size:
        raise DownloadValidationError(f"size mismatch for {dest.name}: expected {expected_size}, got {path.stat().st_size}")
    if expected_sha256:
        actual = _sha256(path)
        if actual != expected_sha256:
            raise DownloadValidationError(f"sha256 mismatch for {dest.name}: expected {expected_sha256}, got {actual}")


def _download_file_urllib(url, tmp, dest, expected_size=None, expected_sha256="", timeout=30):
    """Download file urllib.

    Args:
        url (str): Url value.
        tmp (Any): Tmp value.
        dest (Any): Dest value.
        expected_size (Any, optional): Expected size value. Defaults to None.
        expected_sha256 (Any, optional): Expected sha256 value. Defaults to ''.
        timeout (int, optional): Network timeout in seconds. Defaults to 30.

    Returns:
        Any: Computed result."""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        total = int(response.headers.get("content-length") or expected_size or 0)
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as progress:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                progress.update(len(chunk))
    _validate_downloaded_file(tmp, dest, expected_size, expected_sha256)
    os.replace(tmp, dest)
    return dest


def _download_file_aria2(url, tmp, dest, expected_size=None, expected_sha256="", timeout=30):
    """Download file aria2.

    Args:
        url (str): Url value.
        tmp (Any): Tmp value.
        dest (Any): Dest value.
        expected_size (Any, optional): Expected size value. Defaults to None.
        expected_sha256 (Any, optional): Expected sha256 value. Defaults to ''.
        timeout (int, optional): Network timeout in seconds. Defaults to 30.

    Returns:
        Any: Computed result."""
    cmd = [
        ARIA2C_PATH,
        "--allow-overwrite=true",
        "--auto-file-renaming=false",
        "--continue=true",
        "--console-log-level=warn",
        "--summary-interval=1",
        "--max-connection-per-server=16",
        "--split=16",
        "--min-split-size=1M",
        "--max-tries=3",
        f"--connect-timeout={timeout}",
        f"--timeout={timeout}",
        "--dir",
        str(tmp.parent),
        "--out",
        tmp.name,
        url,
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise DownloadError(f"aria2c failed with exit code {result.returncode}")
    if not tmp.is_file():
        raise DownloadError("aria2c did not create the expected output file")

    _validate_downloaded_file(tmp, dest, expected_size, expected_sha256)
    os.replace(tmp, dest)
    return dest


def _download_file(url, dest, expected_size=None, expected_sha256="", timeout=30, retries=2):
    """Download file.

    Args:
        url (str): Url value.
        dest (Any): Dest value.
        expected_size (Any, optional): Expected size value. Defaults to None.
        expected_sha256 (Any, optional): Expected sha256 value. Defaults to ''.
        timeout (int, optional): Network timeout in seconds. Defaults to 30.
        retries (int, optional): Retries value. Defaults to 2.

    Returns:
        Any: Computed result."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    last_error = None

    for attempt in range(retries + 1):
        try:
            if ARIA2C_PATH:
                return _download_file_aria2(url, tmp, dest, expected_size, expected_sha256, timeout=timeout)
            return _download_file_urllib(url, tmp, dest, expected_size, expected_sha256, timeout=timeout)
        except (OSError, urllib.error.URLError, urllib.error.HTTPError, DownloadError) as exc:
            last_error = exc
            # don't rm temp file if aria2c is being used
            if not ARIA2C_PATH or isinstance(exc, DownloadValidationError):
                _cleanup_partial_download(tmp)
            if attempt < retries:
                time.sleep(1.0 + attempt)
    raise DownloadError(f"failed to download {url}: {last_error}")


def files_for_model(model_name, model_dir=None):
    """Return the local file targets required by a catalog model.

    Args:
        model_name (str): Model name or alias from the pymss catalog.
        model_dir (str | os.PathLike | None, optional): Local model cache directory. Uses the package default when None. Defaults to None.

    Returns:
        tuple[ModelEntry, list[tuple[str, Path]]]: Catalog entry and required files."""
    entry = get_model_entry(model_name)
    files = [(entry.relpath, model_path_for(entry, model_dir))]
    config_path = config_path_for(entry, model_dir)
    if entry.config_relpath and config_path is not None:
        files.append((entry.config_relpath, config_path))
    files.extend(zip(entry.auxiliary_relpaths, auxiliary_paths_for(entry, model_dir)))
    return entry, files


def download_model(model_name, model_dir=None, source="modelscope", endpoint=None, verify=True, force=False, timeout=30):
    """Download all files required by one catalog model.

    The downloader resolves the model from the pymss catalog, downloads the
    weights, config, and auxiliary files, and skips files that already match
    available size/hash metadata. If ``aria2c`` is available on ``PATH``, pymss
    uses it for resumable multi-connection downloads; otherwise it falls back
    to Python's urllib downloader.

    Args:
        model_name (str): Model name, stem, or alias from the pymss catalog.
        model_dir (str | os.PathLike | None, optional): Local model cache
            directory. When omitted, pymss uses its default model directory.
            Defaults to None.
        source (str, optional): Download source. Supported values are
            ``modelscope``, ``huggingface``, and ``hf-mirror``. Defaults to
            ``"modelscope"``.
        endpoint (str | None, optional): Custom endpoint prefix. When provided,
            the final URL is ``endpoint/relative/catalog/path`` and ``source``
            is ignored for URL construction. Defaults to None.
        verify (bool, optional): Whether to validate downloads with available
            ModelScope size/hash metadata. Custom endpoints skip source index
            lookup. Defaults to True.
        force (bool, optional): Whether to redownload files even when existing
            local files appear valid. Defaults to False.
        timeout (int, optional): Network timeout in seconds. Defaults to 30.

    Returns:
        dict: Download result with ``entry`` (``ModelEntry``), ``downloaded``
        (list of paths written this call), and ``skipped`` (list of existing
        valid paths).

    Raises:
        KeyError: If ``model_name`` is unknown.
        DownloadError: If downloading fails after retries.
        DownloadValidationError: If a downloaded file fails size/hash checks.

    Example:
        >>> from pymss import download_model
        >>> result = download_model("bs_roformer_voc_hyperacev2", model_dir="models")
        >>> result["entry"].name

    Example:
        >>> download_model(
        ...     "bs_roformer_voc_hyperacev2",
        ...     source="hf-mirror",
        ...     force=True,
        ...     timeout=60,
        ... )"""
    entry, files = files_for_model(model_name, model_dir)
    index = fetch_modelscope_file_index(timeout=timeout) if verify and endpoint is None else None
    downloaded = []
    skipped = []

    for relpath, dest in files:
        expected_size, expected_sha256 = _expected_size_and_hash(relpath, index)
        if not force and _already_valid(dest, expected_size, expected_sha256):
            skipped.append(str(dest))
            continue
        url = remote_url(relpath, source=source, endpoint=endpoint)
        _download_file(url, dest, expected_size, expected_sha256, timeout=timeout)
        downloaded.append(str(dest))

    return {"entry": entry, "downloaded": downloaded, "skipped": skipped}


def download_all(model_dir=None, source="modelscope", endpoint=None, supported_only=False, force=False, timeout=30):
    """Download every catalog model, optionally limited to supported entries.

    Args:
        model_dir (str | os.PathLike | None, optional): Local model cache directory. Uses the package default when None. Defaults to None.
        source (str, optional): Download source name. Defaults to "modelscope".
        endpoint (str | None, optional): Optional custom download endpoint. Defaults to None.
        supported_only (bool, optional): Supported only value. Defaults to False.
        force (bool, optional): Whether to overwrite or redownload existing files. Defaults to False.
        timeout (int, optional): Network timeout in seconds. Defaults to 30.

    Returns:
        list[dict]: Per-model download results."""
    from .model_registry import list_models

    results = []
    for entry in list_models(supported=True if supported_only else None):
        try:
            results.append(
                download_model(entry.name, model_dir=model_dir, source=source, endpoint=endpoint, force=force, timeout=timeout)
            )
        except Exception as exc:
            results.append({"entry": entry, "error": str(exc)})
    return results
