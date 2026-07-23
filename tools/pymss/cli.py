import argparse
import json
import sys
import warnings

from .ensemble import ENSEMBLE_ALGORITHMS, save_ensemble_audio
from .logger import get_separation_logger
from .model_download import download_all, download_model
from .model_registry import create_separator, list_models, resolve_model
from .progress import _CliInferenceProgress
from .workflow import load_workflow_file, run_workflow_file, validate_workflow, write_workflow_template

warnings.filterwarnings("ignore", category=UserWarning)


def _parse_key_value(values):
    """Parse key value.

    Args:
        values (Any): Values value.

    Returns:
        Any: Parsed value."""
    result = {}
    for value in values or []:
        if "=" not in value:
            raise argparse.ArgumentTypeError(f"Expected key=value, got {value!r}")
        key, raw = value.split("=", 1)
        lowered = raw.lower()
        if lowered in {"true", "false"}:
            result[key] = lowered == "true"
        else:
            try:
                result[key] = int(raw)
            except ValueError:
                try:
                    result[key] = float(raw)
                except ValueError:
                    result[key] = raw
    return result


def cmd_list(args):
    """Implement the cmd list helper.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        Any: Computed result."""
    rows = list_models(category=args.category, supported=None if args.all else True)
    if args.json:
        print(json.dumps([item.__dict__ for item in rows], ensure_ascii=False, indent=2))
        return 0
    for item in rows:
        status = "ok" if item.supported else item.unsupported_reason
        category = item.category_path or item.primary_category
        print(f"{item.name}\t{item.model_type or item.architecture}\t{category}\t{item.target_stem}\t{status}")
    return 0


def cmd_info(args):
    """Implement the cmd info helper.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        Any: Computed result."""
    resolved = resolve_model(args.model, model_dir=args.model_dir, require_supported=False, require_exists=False)
    entry = resolved["entry"]
    data = {
        "name": entry.name,
        "model_type": entry.model_type,
        "architecture": entry.architecture,
        "supported": entry.supported,
        "unsupported_reason": entry.unsupported_reason,
        "category": entry.category_path or entry.primary_category,
        "category_cn": " / ".join(filter(None, [entry.primary_category_cn, entry.secondary_category_cn])),
        "target_stem": entry.target_stem,
        "model_path": resolved["model_path"],
        "config_path": resolved["config_path"],
        "size_bytes": entry.size_bytes,
    }
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


def cmd_download(args):
    """Implement the cmd download helper.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        Any: Computed result."""
    if args.model == "all":
        results = download_all(
            model_dir=args.model_dir,
            source=args.source,
            endpoint=args.endpoint,
            supported_only=args.supported_only,
            force=args.force,
        )
        failed = [item for item in results if item.get("error")]
        print(f"Downloaded/skipped {len(results) - len(failed)} model(s), failed {len(failed)}.")
        for item in failed:
            print(f"ERROR {item['entry'].name}: {item['error']}", file=sys.stderr)
        return 1 if failed else 0

    result = download_model(
        args.model,
        model_dir=args.model_dir,
        source=args.source,
        endpoint=args.endpoint,
        force=args.force,
    )
    _print_download_result(result)
    return 0


def _print_download_result(result):
    """Print download result.

    Args:
        result (Any): Result value.

    Returns:
        None: This callable completes for its side effects."""
    for path in result["skipped"]:
        print(f"exists {path}")
    for path in result["downloaded"]:
        print(f"downloaded {path}")


def _ensure_model_files(args):
    """Ensure model files.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None: This callable completes for its side effects."""
    try:
        resolve_model(args.model, model_dir=args.model_dir, require_supported=True, require_exists=True)
    except FileNotFoundError:
        result = download_model(args.model, model_dir=args.model_dir, source=args.source, endpoint=args.endpoint)
        _print_download_result(result)
    else:
        if args.download:
            result = download_model(args.model, model_dir=args.model_dir, source=args.source, endpoint=args.endpoint)
            _print_download_result(result)


def cmd_infer(args):
    """Implement the cmd infer helper.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        Any: Computed result."""
    _ensure_model_files(args)
    logger = get_separation_logger()
    inference_progress = _CliInferenceProgress()
    with create_separator(
        args.model,
        model_dir=args.model_dir,
        device=args.device,
        device_ids=args.device_ids or [0],
        output_format=args.output_format,
        audio_params={
            "wav_bit_depth": args.wav_bit_depth,
            "flac_bit_depth": args.flac_bit_depth,
            "mp3_bit_rate": args.mp3_bit_rate,
            "m4a_bit_rate": args.m4a_bit_rate,
            "m4a_aac_at_quality": args.m4a_aac_at_quality,
        },
        use_tta=args.tta,
        store_dirs=args.output,
        save_as_folder=args.save_as_folder,
        logger=logger,
        debug=args.debug,
        progress_callback=inference_progress,
        inference_params=_parse_key_value(args.param),
    ) as separator:
        try:
            files = separator.process_folder(args.input)
        finally:
            inference_progress.close()
    logger.info(f"Processed {len(files)} file(s).")
    return 0


def cmd_ensemble(args):
    """Run the ensemble CLI command.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        Any: Computed result."""
    logger = get_separation_logger()
    output_path = save_ensemble_audio(
        args.files,
        args.output,
        algorithm=args.algorithm,
        weights=args.weights,
        output_format=args.output_format,
        audio_params={
            "wav_bit_depth": args.wav_bit_depth,
            "flac_bit_depth": args.flac_bit_depth,
            "mp3_bit_rate": args.mp3_bit_rate,
            "m4a_bit_rate": args.m4a_bit_rate,
            "m4a_codec": args.m4a_codec,
            "m4a_aac_at_quality": args.m4a_aac_at_quality,
        },
        logger=logger,
    )
    logger.info(f"Saved ensemble audio to {output_path}")
    return 0


def cmd_workflow_init(args):
    """Write a starter workflow file."""
    path = write_workflow_template(args.output, overwrite=args.force)
    print(f"Wrote workflow template to {path}")
    return 0


def cmd_workflow_validate(args):
    """Validate a workflow file without running inference."""
    workflow = load_workflow_file(args.config)
    model_resolver = resolve_model if args.check_models or args.require_files else None
    validate_workflow(
        workflow,
        model_dir=args.model_dir,
        require_model_files=args.require_files,
        model_resolver=model_resolver,
    )
    print(f"Workflow is valid: {len(workflow.steps)} step(s).")
    return 0


def cmd_workflow_run(args):
    """Run an audio workflow from a YAML/JSON file."""
    logger = get_separation_logger()
    files = run_workflow_file(
        args.config,
        args.input,
        args.output,
        model_dir=args.model_dir,
        device=args.device,
        output_format=args.output_format,
        download=args.download,
        source=args.source,
        endpoint=args.endpoint,
        output_layout=args.output_layout,
        audio_params={
            "wav_bit_depth": args.wav_bit_depth,
            "flac_bit_depth": args.flac_bit_depth,
            "mp3_bit_rate": args.mp3_bit_rate,
            "m4a_bit_rate": args.m4a_bit_rate,
            "m4a_codec": args.m4a_codec,
            "m4a_aac_at_quality": args.m4a_aac_at_quality,
        },
        logger=logger,
        debug=args.debug,
    )
    logger.info(f"Processed {len(files)} file(s).")
    return 0


def cmd_serve(args):
    """Implement the cmd serve helper.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        Any: Computed result."""
    from .server import ServerConfig, run_server

    config = ServerConfig(
        model=args.model,
        model_dir=args.model_dir,
        source=args.source,
        endpoint=args.endpoint,
        device=args.device,
        device_ids=args.device_ids or [0],
        api_key=args.api_key,
        host=args.host,
        port=args.port,
        debug=args.debug,
        inference_params=_parse_key_value(args.param),
        max_audio_seconds=args.max_audio_seconds,
        max_request_bytes=args.max_request_bytes,
        max_queue_size=args.max_queue_size,
        request_timeout_seconds=args.request_timeout_seconds,
        webui=args.webui,
    )
    run_server(config)
    return 0


def build_parser():
    """Build the pymss command-line parser.

    Args:
        None: This callable does not accept user-provided arguments.

    Returns:
        argparse.ArgumentParser: Configured CLI parser."""
    parser = argparse.ArgumentParser(
        prog="pymss",
        description="Command-line interface for the pymss music source separation package.",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=60),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ==========================
    # List models
    # ==========================
    list_parser = subparsers.add_parser(
        "list",
        help="List known models.",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=60),
    )
    list_parser.add_argument("--category", help="Filter by primary or secondary category.")
    list_parser.add_argument("--all", action="store_true", help="Include models that are not supported for inference yet.")
    list_parser.add_argument("--json", action="store_true")
    list_parser.set_defaults(func=cmd_list)

    # ==========================
    # Show model info
    # ==========================
    info_parser = subparsers.add_parser(
        "info",
        help="Show model metadata.",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=60),
    )
    info_parser.add_argument("model")
    info_parser.add_argument(
        "--model-dir",
        help="Local model cache directory. Defaults to PYMSS_MODEL_DIR, repository all_models if present, or ~/.cache/pymss/models.",
    )
    info_parser.set_defaults(func=cmd_info)

    # ==========================
    # Download models
    # ==========================
    download_parser = subparsers.add_parser(
        "download",
        help="Download a model by name, or use 'all'.",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=60),
    )
    download_parser.add_argument("model")
    download_parser.add_argument(
        "--model-dir",
        help="Local model cache directory. Defaults to PYMSS_MODEL_DIR, repository all_models if present, or ~/.cache/pymss/models.",
    )
    download_parser.add_argument("--source", default="modelscope", choices=["modelscope", "huggingface", "hf-mirror"])
    download_parser.add_argument("--endpoint", help="Custom resolve endpoint. It must serve files by relative path.")
    download_parser.add_argument("--force", action="store_true")
    download_parser.add_argument("--supported-only", action="store_true", help="Only used with model='all'.")
    download_parser.set_defaults(func=cmd_download)

    # ==========================
    # Inference
    # ==========================
    infer_parser = subparsers.add_parser(
        "infer",
        help="Run inference by model name.",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=60),
    )
    infer_parser.add_argument("model")
    infer_parser.add_argument(
        "--model-dir",
        help="Local model cache directory. Defaults to PYMSS_MODEL_DIR, repository all_models if present, or ~/.cache/pymss/models.",
    )
    infer_parser.add_argument(
        "--download",
        action="store_true",
        help="Check/download the model before inference. Missing model files are downloaded automatically.",
    )
    infer_parser.add_argument("--source", default="modelscope", choices=["modelscope", "huggingface", "hf-mirror"])
    infer_parser.add_argument("--endpoint", help="Custom resolve endpoint. It must serve files by relative path.")
    infer_parser.add_argument("-i", "--input", required=True, help="Input audio file or folder.")
    infer_parser.add_argument("-o", "--output", default="results", help="Output folder.")
    infer_parser.add_argument(
        "--save-as-folder",
        action="store_true",
        help="Save each input audio file's separated stems in a subfolder named after the audio file.",
    )
    infer_parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps", "mlx"])
    infer_parser.add_argument(
        "--device-id", action="append", type=int, dest="device_ids", help="CUDA device id. Can be repeated."
    )
    infer_parser.add_argument("--format", default="wav", choices=["wav", "flac", "mp3", "m4a"], dest="output_format")
    infer_parser.add_argument("--wav-bit-depth", default="FLOAT", choices=["FLOAT", "PCM_16", "PCM_24"])
    infer_parser.add_argument("--flac-bit-depth", default="PCM_16", choices=["PCM_16", "PCM_24"])
    infer_parser.add_argument("--mp3-bit-rate", default="320k")
    infer_parser.add_argument("--m4a-bit-rate", default="512k")
    infer_parser.add_argument("--m4a-codec", default="aac")
    infer_parser.add_argument("--m4a-aac-at-quality", default=2, type=int)
    infer_parser.add_argument("--tta", action="store_true", help="Enable test time augmentation.")
    infer_parser.add_argument("--debug", action="store_true")
    infer_parser.add_argument(
        "--param", action="append", default=[], help="Inference override as key=value, for example --param batch_size=2."
    )
    infer_parser.set_defaults(func=cmd_infer)

    # ==========================
    # Ensemble
    # ==========================
    ensemble_parser = subparsers.add_parser(
        "ensemble",
        help="Combine multiple audio files with an ensemble algorithm.",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=60),
    )
    ensemble_parser.add_argument("files", nargs="+", help="Input audio files. At least two files are required.")
    ensemble_parser.add_argument(
        "-a",
        "--algorithm",
        default="avg_wave",
        choices=ENSEMBLE_ALGORITHMS,
        help="Ensemble algorithm.",
    )
    ensemble_parser.add_argument(
        "-w",
        "--weights",
        nargs="+",
        type=float,
        help="Input weights, for example --weights 1 0.8 1.2. Defaults to all 1.",
    )
    ensemble_parser.add_argument("-o", "--output", required=True, help="Output audio file.")
    ensemble_parser.add_argument("--format", choices=["wav", "flac", "mp3", "m4a"], dest="output_format")
    ensemble_parser.add_argument("--wav-bit-depth", default="FLOAT", choices=["FLOAT", "PCM_16", "PCM_24"])
    ensemble_parser.add_argument("--flac-bit-depth", default="PCM_16", choices=["PCM_16", "PCM_24"])
    ensemble_parser.add_argument("--mp3-bit-rate", default="320k")
    ensemble_parser.add_argument("--m4a-bit-rate", default="512k")
    ensemble_parser.add_argument("--m4a-codec", default="aac")
    ensemble_parser.add_argument("--m4a-aac-at-quality", default=2, type=int)
    ensemble_parser.set_defaults(func=cmd_ensemble)

    # ==========================
    # Workflow
    # ==========================
    workflow_parser = subparsers.add_parser(
        "workflow",
        help="Create, validate, or run an automatic multi-model workflow.",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=60),
    )
    workflow_subparsers = workflow_parser.add_subparsers(dest="workflow_command", required=True)

    workflow_init_parser = workflow_subparsers.add_parser(
        "init",
        help="Write a starter workflow YAML file.",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=60),
    )
    workflow_init_parser.add_argument("-o", "--output", default="workflow.yaml", help="Workflow file to create.")
    workflow_init_parser.add_argument("--force", action="store_true", help="Overwrite the output file if it exists.")
    workflow_init_parser.set_defaults(func=cmd_workflow_init)

    workflow_validate_parser = workflow_subparsers.add_parser(
        "validate",
        help="Validate a workflow YAML/JSON file.",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=60),
    )
    workflow_validate_parser.add_argument("-c", "--config", required=True, help="Workflow YAML/JSON file.")
    workflow_validate_parser.add_argument(
        "--model-dir",
        help="Local model cache directory used when --require-files is set.",
    )
    workflow_validate_parser.add_argument(
        "--check-models",
        action="store_true",
        help="Also check that every referenced model exists in the catalog.",
    )
    workflow_validate_parser.add_argument(
        "--require-files",
        action="store_true",
        help="Also require every referenced catalog model file to exist locally.",
    )
    workflow_validate_parser.set_defaults(func=cmd_workflow_validate)

    workflow_run_parser = workflow_subparsers.add_parser(
        "run",
        help="Run inference through a workflow YAML/JSON file.",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=60),
    )
    workflow_run_parser.add_argument("-c", "--config", required=True, help="Workflow YAML/JSON file.")
    workflow_run_parser.add_argument("-i", "--input", required=True, help="Input audio file or folder.")
    workflow_run_parser.add_argument("-o", "--output", default="results", help="Output folder.")
    workflow_run_parser.add_argument(
        "--output-layout",
        default="folders",
        choices=["folders", "flat"],
        help=(
            "Workflow output layout. 'folders' keeps each input under <output>/<audio>/; "
            "'flat' writes outputs directly under the workflow task folder and save subfolders."
        ),
    )
    workflow_run_parser.add_argument(
        "--model-dir",
        help="Local model cache directory. Workflow step model_dir values take precedence.",
    )
    workflow_run_parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing model files before each workflow step is loaded.",
    )
    workflow_run_parser.add_argument("--source", default="modelscope", choices=["modelscope", "huggingface", "hf-mirror"])
    workflow_run_parser.add_argument("--endpoint", help="Custom resolve endpoint. It must serve files by relative path.")
    workflow_run_parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps", "mlx"])
    workflow_run_parser.add_argument("--format", choices=["wav", "flac", "mp3", "m4a"], dest="output_format")
    workflow_run_parser.add_argument("--wav-bit-depth", default="FLOAT", choices=["FLOAT", "PCM_16", "PCM_24"])
    workflow_run_parser.add_argument("--flac-bit-depth", default="PCM_16", choices=["PCM_16", "PCM_24"])
    workflow_run_parser.add_argument("--mp3-bit-rate", default="320k")
    workflow_run_parser.add_argument("--m4a-bit-rate", default="512k")
    workflow_run_parser.add_argument("--m4a-codec", default="aac")
    workflow_run_parser.add_argument("--m4a-aac-at-quality", default=2, type=int)
    workflow_run_parser.add_argument("--debug", action="store_true")
    workflow_run_parser.set_defaults(func=cmd_workflow_run)

    # ==========================
    # Server
    # ==========================
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start an OpenAI-style HTTP inference server.",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=60),
    )
    serve_parser.add_argument("model", nargs="?")
    serve_parser.add_argument(
        "--model-dir",
        help="Local model cache directory. Defaults to PYMSS_MODEL_DIR, repository all_models if present, or ~/.cache/pymss/models.",
    )
    serve_parser.add_argument("--source", default="modelscope", choices=["modelscope", "huggingface", "hf-mirror"])
    serve_parser.add_argument("--endpoint", help="Custom resolve endpoint. It must serve files by relative path.")
    serve_parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps", "mlx"])
    serve_parser.add_argument(
        "--device-id",
        action="append",
        type=int,
        dest="device_ids",
        help="CUDA device id. Can be repeated.",
    )
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", default=8000, type=int)
    serve_parser.add_argument("--api-key", help="Optional bearer token required for /v1/* endpoints.")
    serve_parser.add_argument("--debug", action="store_true")
    serve_parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Inference override as key=value, for example --param batch_size=2.",
    )
    serve_parser.add_argument("--max-audio-seconds", default=600.0, type=float)
    serve_parser.add_argument("--max-request-bytes", default=536870912, type=int)
    serve_parser.add_argument("--max-queue-size", default=8, type=int)
    serve_parser.add_argument("--request-timeout-seconds", default=0.0, type=float)
    serve_parser.add_argument("--webui", action="store_true", help="Serve the optional browser WebUI at /ui/.")
    serve_parser.set_defaults(func=cmd_serve)

    return parser


def main(argv=None):
    """Run the pymss command-line interface.

    Args:
        argv (Sequence[str] | None, optional): Command-line arguments. Uses sys.argv when None. Defaults to None.

    Returns:
        int: Process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        print(f"pymss: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
