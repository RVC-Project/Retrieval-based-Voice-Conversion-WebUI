"""CLI entrypoint for RVC evaluation metrics."""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tools.eval.metrics.f0_accuracy import compute_f0_rmse
from tools.eval.metrics.mcd import compute_mcd

try:
    from tools.eval.metrics.whisper_cer import compute_whisper_cer

    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

THRESHOLDS = {
    "mcd": {"pass": 6.0, "fail": 8.0, "unit": "dB", "lower_is_better": True},
    "f0_rmse": {"pass": 20.0, "fail": 50.0, "unit": "cents", "lower_is_better": True},
    "whisper_cer": {"pass": 0.10, "fail": 0.20, "unit": "ratio", "lower_is_better": True},
}

VALID_METRICS = {"all", "mcd", "f0", "cer"}


def _judge(value: float, threshold: dict) -> str:
    """Return PASS / WARN / FAIL based on threshold dict."""
    if threshold["lower_is_better"]:
        if value <= threshold["pass"]:
            return "PASS"
        if value >= threshold["fail"]:
            return "FAIL"
        return "WARN"
    else:
        if value >= threshold["pass"]:
            return "PASS"
        if value <= threshold["fail"]:
            return "FAIL"
        return "WARN"


def _worst_status(statuses: list[str]) -> str:
    """Return the worst status among a list of PASS/WARN/FAIL."""
    order = {"PASS": 0, "WARN": 1, "FAIL": 2}
    worst = max(statuses, key=lambda s: order[s])
    return worst


def _load_config(config_path: str) -> dict:
    """Load mel parameters from a model config JSON."""
    with open(config_path) as f:
        cfg = json.load(f)
    data = cfg.get("data", {})
    return {
        "sampling_rate": data.get("sampling_rate", 48000),
        "hop_length": data.get("hop_length", 480),
        "n_mel_channels": data.get("n_mel_channels", 128),
        "mel_fmin": data.get("mel_fmin", 0.0),
        "mel_fmax": data.get("mel_fmax", None),
    }


def _detect_device() -> str:
    """Auto-detect cuda/cpu."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="RVC evaluation: compute quality metrics between reference and converted audio.",
    )
    parser.add_argument("--ref", required=True, help="Path to reference audio WAV")
    parser.add_argument("--conv", required=True, help="Path to converted audio WAV")
    parser.add_argument(
        "--metrics", default="all",
        help="Comma-separated metrics to compute. Choices: all, mcd, f0, cer (default: all)",
    )
    parser.add_argument("--whisper-model", default="large-v3", help="Whisper model size (default: large-v3)")
    parser.add_argument("--whisper-lang", default="ja", help="Whisper language (default: ja)")
    parser.add_argument("--ref-text", default=None, help="Reference text for CER computation")
    parser.add_argument(
        "--config", default="configs/v2/48k.json",
        help="Model config JSON path (default: configs/v2/48k.json)",
    )
    parser.add_argument("--device", default=None, help="Device: cuda or cpu (default: auto-detect)")
    parser.add_argument("--output", default=None, help="JSON output file path (default: stdout)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run evaluation metrics and output results as JSON."""
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Validate input files
    for label, path in [("reference", args.ref), ("converted", args.conv)]:
        if not os.path.isfile(path):
            logger.error("%s file not found: %s", label, path)
            sys.exit(1)

    # Parse requested metrics
    requested = {m.strip() for m in args.metrics.split(",")}
    invalid = requested - VALID_METRICS
    if invalid:
        logger.error("Unknown metrics: %s. Valid: %s", invalid, VALID_METRICS)
        sys.exit(1)
    run_all = "all" in requested
    run_mcd = run_all or "mcd" in requested
    run_f0 = run_all or "f0" in requested
    run_cer = run_all or "cer" in requested

    # Load config
    config_path = args.config
    if not os.path.isabs(config_path):
        # Resolve relative to project root (three levels up from this file)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        config_path = os.path.join(project_root, config_path)

    if os.path.isfile(config_path):
        mel_params = _load_config(config_path)
        logger.debug("Loaded config from %s: %s", config_path, mel_params)
    else:
        logger.warning("Config file not found: %s, using defaults", config_path)
        mel_params = {
            "sampling_rate": 48000,
            "hop_length": 480,
            "n_mel_channels": 128,
            "mel_fmin": 0.0,
            "mel_fmax": None,
        }

    device = args.device if args.device else _detect_device()
    logger.info("Using device: %s", device)

    sr = mel_params["sampling_rate"]
    hop = mel_params["hop_length"]
    fmin = mel_params["mel_fmin"]
    fmax = mel_params["mel_fmax"]

    metrics_result = {}
    statuses = []

    # --- MCD ---
    if run_mcd:
        logger.info("Computing MCD...")
        result = compute_mcd(
            ref_path=args.ref, conv_path=args.conv,
            sr=sr, fmin=fmin, fmax=fmax, hop_length=hop,
        )
        status = _judge(result["value"], THRESHOLDS["mcd"])
        statuses.append(status)
        metrics_result["mcd"] = {
            **result,
            "status": status,
            "thresholds": THRESHOLDS["mcd"],
        }
        logger.info("MCD: %.2f %s [%s]", result["value"], result["unit"], status)

    # --- F0 RMSE ---
    if run_f0:
        logger.info("Computing F0 RMSE...")
        result = compute_f0_rmse(
            ref_path=args.ref, conv_path=args.conv,
            sr=sr, hop_length=hop, device=device,
        )
        status = _judge(result["value"], THRESHOLDS["f0_rmse"])
        statuses.append(status)
        metrics_result["f0_rmse"] = {
            **result,
            "status": status,
            "thresholds": THRESHOLDS["f0_rmse"],
        }
        logger.info("F0 RMSE: %.2f %s [%s]", result["value"], result["unit"], status)

    # --- Whisper CER ---
    if run_cer:
        if not HAS_WHISPER:
            logger.warning("whisper_cer module not available, skipping CER metric")
        else:
            logger.info("Computing Whisper CER...")
            result = compute_whisper_cer(
                ref_path=args.ref, conv_path=args.conv,
                ref_text=args.ref_text,
                model_name=args.whisper_model,
                language=args.whisper_lang,
                device=device,
            )
            status = _judge(result["value"], THRESHOLDS["whisper_cer"])
            statuses.append(status)
            metrics_result["whisper_cer"] = {
                **result,
                "status": status,
                "thresholds": THRESHOLDS["whisper_cer"],
            }
            logger.info("Whisper CER: %.4f %s [%s]", result["value"], result["unit"], status)

    # Overall status
    overall = _worst_status(statuses) if statuses else "PASS"

    output = {
        "version": "0.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "files": {
            "reference": os.path.abspath(args.ref),
            "converted": os.path.abspath(args.conv),
        },
        "metrics": metrics_result,
        "overall_status": overall,
        "config": {
            "whisper_model": args.whisper_model,
            "whisper_lang": args.whisper_lang,
            "sample_rate": sr,
            "config_file": args.config,
        },
    }

    json_str = json.dumps(output, indent=2, ensure_ascii=False)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_str)
        logger.info("Results written to %s", args.output)
    else:
        print(json_str)


if __name__ == "__main__":
    main()
