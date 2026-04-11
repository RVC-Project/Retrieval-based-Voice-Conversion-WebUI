"""RVC inference latency measurement."""

import logging
import os
import statistics
import sys
import time

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def compute_latency(
    model_path: str,
    audio_path: str,
    index_path: str | None = None,
    f0_method: str = "rmvpe",
    f0_up_key: int = 0,
    device: str = "cuda",
    n_runs: int = 3,
    warmup: int = 1,
) -> dict:
    """Measure RVC inference latency.

    Loads the model via the VC class and runs ``vc_single`` repeatedly,
    recording wall-clock time for each run.  Warmup runs are excluded
    from the reported statistics.

    Args:
        model_path: Path to .pth model file (absolute or relative).
        audio_path: Path to input audio file.
        index_path: Optional path to FAISS index file.
        f0_method: F0 extraction method (e.g. ``rmvpe``, ``fcpe``).
        f0_up_key: Pitch shift in semitones.
        device: ``cuda`` or ``cpu``.
        n_runs: Number of measurement runs (excluding warmup).
        warmup: Number of warmup runs.

    Returns:
        dict with keys:
            value: median latency in ms
            unit: ``"ms"``
            details:
                rtf: real-time factor (inference_time / audio_duration).
                     RTF < 1.0 means faster than real-time.
                audio_duration_s: duration of input audio in seconds
                latencies_ms: list of individual run latencies
                n_runs: number of measurement runs
                device: device used

    Raises:
        FileNotFoundError: If *model_path* or *audio_path* does not exist.
        RuntimeError: If inference fails on all attempts.
    """
    import librosa
    import torch

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    model_path = os.path.abspath(model_path)
    audio_path = os.path.abspath(audio_path)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # ------------------------------------------------------------------
    # Prepare environment for VC / Config
    # ------------------------------------------------------------------
    # Config.arg_parse() calls argparse on sys.argv.  Save and reset
    # sys.argv so it doesn't collide with the caller's CLI arguments.
    saved_argv = sys.argv
    sys.argv = [sys.argv[0]]

    # VC.get_vc() constructs the full path as
    #   f"{os.getenv('weight_root')}/{sid}"
    # so we must set weight_root to the directory and pass the basename.
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path)
    os.environ["weight_root"] = model_dir

    from dotenv import load_dotenv

    load_dotenv()

    # Import after sys.argv reset so Config.arg_parse() succeeds.
    from configs.config import Config
    from infer.modules.vc.modules import VC

    try:
        config = Config()
        # Override device if requested and available.
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to cpu")
            device = "cpu"
        if device == "cpu":
            config.device = "cpu"
            config.is_half = False

        vc = VC(config)
        vc.get_vc(model_name)
    finally:
        sys.argv = saved_argv

    # ------------------------------------------------------------------
    # Audio duration (for RTF calculation)
    # ------------------------------------------------------------------
    audio_duration_s = librosa.get_duration(filename=audio_path)
    logger.info(
        "Audio duration: %.2f s, device: %s, warmup: %d, n_runs: %d",
        audio_duration_s,
        device,
        warmup,
        n_runs,
    )

    # ------------------------------------------------------------------
    # Inference helper
    # ------------------------------------------------------------------
    def _run_inference() -> str:
        """Run a single inference pass and return the info string."""
        info, _wav = vc.vc_single(
            0,                          # sid (speaker id)
            audio_path,                 # input_audio_path
            f0_up_key,                  # f0_up_key
            None,                       # f0_file
            f0_method,                  # f0_method
            index_path or "",           # file_index
            "",                         # file_index2
            0.75,                       # index_rate
            3,                          # filter_radius
            0,                          # resample_sr
            1,                          # rms_mix_rate
            0.33,                       # protect
        )
        return info

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------
    for i in range(warmup):
        logger.debug("Warmup run %d/%d", i + 1, warmup)
        info = _run_inference()
        if "Success" not in info:
            raise RuntimeError(f"Inference failed during warmup: {info}")

    # Synchronise GPU before measurement
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Measurement runs
    # ------------------------------------------------------------------
    latencies_ms: list[float] = []
    for i in range(n_runs):
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        info = _run_inference()

        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)

        if "Success" not in info:
            raise RuntimeError(f"Inference failed on run {i + 1}: {info}")

        logger.info("Run %d/%d: %.1f ms", i + 1, n_runs, elapsed_ms)

    # ------------------------------------------------------------------
    # Aggregate results
    # ------------------------------------------------------------------
    median_ms = statistics.median(latencies_ms)
    rtf = (median_ms / 1000.0) / audio_duration_s if audio_duration_s > 0 else float("inf")

    logger.info(
        "Median latency: %.1f ms, RTF: %.3f (%s)",
        median_ms,
        rtf,
        "real-time OK" if rtf < 1.0 else "slower than real-time",
    )

    return {
        "value": median_ms,
        "unit": "ms",
        "details": {
            "rtf": rtf,
            "audio_duration_s": audio_duration_s,
            "latencies_ms": latencies_ms,
            "n_runs": n_runs,
            "device": device,
        },
    }
