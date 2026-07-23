import gc
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from configs.config import Config
from tools.process_utils import kill_process_tree


logger = logging.getLogger(__name__)
config = Config()
weight_pymss_root = Path(os.getenv("weight_pymss_root", "assets/pymss_weights"))
tools_root = str(Path(__file__).resolve().parent)
if tools_root not in sys.path:
    sys.path.insert(0, tools_root)

MODEL_SAMPLE_RATE = 44100
DML_CHUNK_SIZE = 88200
DML_OVERLAP_SIZE = 22050
DML_MODEL_DTYPE_ENV = "PYMSS_DML_MODEL_DTYPE"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FFMPEG_PATH = PROJECT_ROOT / "ffmpeg.exe"
AUDIO_PARAMS = {
    "wav_bit_depth": "FLOAT",
    "flac_bit_depth": "PCM_24",
    "mp3_bit_rate": "320k",
    "m4a_bit_rate": "320k",
    "m4a_codec": "aac",
    "m4a_aac_at_quality": 2,
}


@dataclass(frozen=True)
class ModelSpec:
    label: str
    model_id: str
    model_type: str
    model_file: str
    config_file: str
    desired_stem: str
    secondary_stem: str
    desired_suffix: str
    secondary_suffix: str
    batch_size: int
    overlap_size: int


MODEL_SPECS = (
    ModelSpec(
        label="去混响",
        model_id="dereverb-less-aggressive-18.8050",
        model_type="mel_band_roformer",
        model_file="dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt",
        config_file="dereverb_mel_band_roformer_anvuew.yaml",
        desired_stem="noreverb",
        secondary_stem="reverb",
        desired_suffix="noreverb",
        secondary_suffix="reverb",
        batch_size=1,
        overlap_size=176400,
    ),
    ModelSpec(
        label="去混响（激进）",
        model_id="dereverb-anvuew-19.1729",
        model_type="mel_band_roformer",
        model_file="dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
        config_file="dereverb_mel_band_roformer_anvuew.yaml",
        desired_stem="noreverb",
        secondary_stem="reverb",
        desired_suffix="noreverb",
        secondary_suffix="reverb",
        batch_size=1,
        overlap_size=176400,
    ),
    ModelSpec(
        label="去伴奏",
        model_id="vocals-bs-roformer-368",
        model_type="bs_roformer",
        model_file="model_bs_roformer_ep_368_sdr_12.9628.ckpt",
        config_file="model_bs_roformer_ep_368_sdr_12.9628.yaml",
        desired_stem="vocals",
        secondary_stem="instrumental",
        desired_suffix="vocals",
        secondary_suffix="instrumental",
        batch_size=1,
        overlap_size=264600,
    ),
    ModelSpec(
        label="去伴奏（激进）",
        model_id="vocals-bs-roformer-317",
        model_type="bs_roformer",
        model_file="model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        config_file="model_bs_roformer_ep_317_sdr_12.9755.yaml",
        desired_stem="vocals",
        secondary_stem="other",
        desired_suffix="vocals",
        secondary_suffix="instrumental",
        batch_size=4,
        overlap_size=176400,
    ),
    ModelSpec(
        label="提主旋律",
        model_id="karaoke-mel-roformer-10.1956",
        model_type="mel_band_roformer",
        model_file="model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
        config_file="config_mel_band_roformer_karaoke.yaml",
        desired_stem="karaoke",
        secondary_stem="other",
        desired_suffix="main_vocal",
        secondary_suffix="off_vocal",
        batch_size=1,
        overlap_size=264600,
    ),
)

MODEL_BY_LABEL = {spec.label: spec for spec in MODEL_SPECS}
MODEL_BY_ID = {spec.model_id: spec for spec in MODEL_SPECS}
PYMSS_MODEL_CHOICES = [spec.label for spec in MODEL_SPECS]
PYMSS_WORKER_STATE_LOCK = threading.Lock()
PYMSS_WORKER_STATE = {
    "task_id": 0,
    "active": False,
    "process": None,
    "stop_requested": False,
}
PYMSS_WORKER_OUTPUT_LOCK = threading.Lock()
DML_FP16_DISABLED_MODEL_TYPES = set()


class DMLFP16Fallback(RuntimeError):
    """Request one retry in a fresh DirectML FP32 worker."""


def _normalize_dml_model_dtype(value):
    normalized = str(value or "auto").strip().lower().replace("torch.", "")
    aliases = {
        "auto": "auto",
        "fp16": "float16",
        "half": "float16",
        "float16": "float16",
        "fp32": "float32",
        "float": "float32",
        "float32": "float32",
    }
    if normalized not in aliases:
        logger.warning(
            "Invalid %s=%r; using auto",
            DML_MODEL_DTYPE_ENV,
            value,
        )
        return "auto"
    return aliases[normalized]


def _dml_model_dtype_attempts(spec):
    requested = _normalize_dml_model_dtype(
        os.getenv(DML_MODEL_DTYPE_ENV, "auto")
    )
    if requested == "float32":
        return ("float32",)
    if requested == "float16":
        return ("float16",)
    if spec.model_type in DML_FP16_DISABLED_MODEL_TYPES:
        return ("float32",)
    return ("float16", "float32")


def resolve_model(model_name):
    if not model_name:
        return MODEL_SPECS[0]
    spec = MODEL_BY_LABEL.get(model_name) or MODEL_BY_ID.get(model_name)
    if spec is None:
        raise ValueError("Unknown separation model: %s" % model_name)
    return spec


def get_model_info(model_name):
    spec = resolve_model(model_name)
    return "%s | %s" % (spec.model_type, spec.model_id)


def clean_path(path):
    path = path or ""
    if path.endswith(("\\", "/")):
        path = path[:-1]
    return path.replace("/", os.sep).replace("\\", os.sep).strip(" '\n\"\u202a")


def _uploaded_path(item):
    if isinstance(item, (str, os.PathLike)):
        return os.fspath(item)
    if isinstance(item, dict):
        return item.get("name") or item.get("path")
    return getattr(item, "name", None)


def collect_input_paths(inp_root, paths):
    inp_root = clean_path(inp_root)
    if inp_root:
        if os.path.isfile(inp_root):
            candidates = [inp_root]
        elif os.path.isdir(inp_root):
            candidates = [os.path.join(inp_root, name) for name in sorted(os.listdir(inp_root))]
        else:
            raise FileNotFoundError(inp_root)
    else:
        candidates = [_uploaded_path(item) for item in (paths or [])]
    return [os.path.abspath(path) for path in candidates if path and os.path.isfile(path)]


def _write_audio(path, audio, sample_rate, output_format):
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        channels = 1
    elif audio.ndim == 2 and audio.shape[1] in (1, 2):
        channels = audio.shape[1]
    else:
        raise ValueError("Unsupported audio shape: %s" % (audio.shape,))

    if output_format == "wav":
        sf.write(path, audio, sample_rate, format="WAV", subtype="FLOAT")
        return
    if output_format == "flac":
        sf.write(path, audio, sample_rate, format="FLAC", subtype="PCM_24")
        return

    ffmpeg = str(FFMPEG_PATH) if FFMPEG_PATH.is_file() else "ffmpeg"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "f32le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-i",
        "pipe:0",
        "-vn",
    ]
    if output_format == "mp3":
        command.extend(("-c:a", "libmp3lame", "-b:a", "320k"))
    elif output_format == "m4a":
        command.extend(("-c:a", "aac", "-aac_coder", "fast", "-b:a", "320k"))
    else:
        raise ValueError("Unsupported output format: %s" % output_format)
    command.append(path)
    completed = subprocess.run(
        command,
        input=audio.tobytes(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError("FFmpeg audio encoding failed: %s" % detail)


class MSSTBatchSeparator:
    def __init__(
        self,
        spec,
        output_format,
        desired_root,
        secondary_root,
        model_dtype="auto",
        progress_callback=None,
        separation_logger=None,
    ):
        try:
            from pymss import MSSeparator, load_audio
        except ImportError as error:
            raise RuntimeError(
                "项目内置的 pymss 运行库加载失败"
            ) from error

        self.spec = spec
        self.output_format = output_format.lower()
        if self.output_format not in {"wav", "flac", "mp3", "m4a"}:
            raise ValueError("Unsupported output format: %s" % output_format)
        desired_root = clean_path(desired_root)
        secondary_root = clean_path(secondary_root)
        if not desired_root or not secondary_root:
            raise ValueError("输出文件夹不能为空")
        self.desired_root = os.path.abspath(desired_root)
        self.secondary_root = os.path.abspath(secondary_root)
        os.makedirs(self.desired_root, exist_ok=True)
        os.makedirs(self.secondary_root, exist_ok=True)

        model_path = weight_pymss_root / spec.model_file
        config_path = weight_pymss_root / spec.config_file
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        if not config_path.is_file():
            raise FileNotFoundError(config_path)

        parsed_device = torch.device(config.device)
        use_cuda = parsed_device.type == "cuda"
        use_dml = parsed_device.type == "privateuseone"
        model_dtype = _normalize_dml_model_dtype(model_dtype) if use_dml else "auto"
        device_id = parsed_device.index if use_cuda and parsed_device.index is not None else 0
        pymss_device = str(parsed_device) if use_dml else ("cuda" if use_cuda else "cpu")
        batch_size = 1 if use_dml else spec.batch_size
        chunk_size = DML_CHUNK_SIZE if use_dml else 352800
        overlap_size = DML_OVERLAP_SIZE if use_dml else spec.overlap_size
        self._load_audio = load_audio
        self.model_load_count = 0
        self.separator = None
        self._save_pool = None
        try:
            self.separator = MSSeparator(
                model_type=spec.model_type,
                model_path=str(model_path),
                config_path=str(config_path),
                device=pymss_device,
                device_ids=[device_id],
                output_format=self.output_format,
                use_tta=False,
                store_dirs={},
                audio_params=AUDIO_PARAMS,
                logger=separation_logger,
                debug=False,
                progress_callback=progress_callback,
                inference_params={
                    "batch_size": batch_size,
                    "chunk_size": chunk_size,
                    "overlap_size": overlap_size,
                    "standardize": False,
                    "normalize": False,
                    "use_amp": bool(config.is_half and use_cuda),
                    "model_dtype": model_dtype,
                    "cuda_attention_backend": "default",
                },
            )
            self.separator.config.training.use_amp = bool(config.is_half and use_cuda)
            model = getattr(self.separator, "model", None)
            model = getattr(model, "module", model)
            parameter = next(model.parameters(), None) if model is not None else None
            self.model_dtype = (
                str(parameter.dtype).replace("torch.", "")
                if parameter is not None
                else "unknown"
            )
            if use_dml and model_dtype == "float16" and self.model_dtype != "float16":
                raise RuntimeError(
                    "DirectML FP16 was requested, but the loaded model uses %s"
                    % self.model_dtype
                )
            self._save_pool = ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="rvc-msst-save",
            )
        except Exception:
            self.close()
            raise
        self.model_load_count = 1
        logger.info(
            "Loaded MSST model once for batch: %s, device=%s, model_dtype=%s, amp=%s",
            spec.model_id,
            self.separator.device,
            self.model_dtype,
            bool(config.is_half and use_cuda),
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback_value):
        self.close()

    def _save_output(self, audio, sample_rate, output_root, file_stem, suffix):
        output_path = os.path.join(
            output_root,
            "%s_%s.%s" % (file_stem, suffix, self.output_format),
        )
        temp_path = os.path.join(
            output_root,
            ".%s_%s.%s.tmp.%s"
            % (file_stem, suffix, uuid.uuid4().hex, self.output_format),
        )
        started = time.perf_counter()
        try:
            _write_audio(temp_path, audio, sample_rate, self.output_format)
            if not os.path.isfile(temp_path) or os.path.getsize(temp_path) == 0:
                raise RuntimeError("音频编码没有生成有效文件: %s" % output_path)
            os.replace(temp_path, output_path)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise
        return output_path, time.perf_counter() - started

    def separate_file(self, input_path):
        mix, sample_rate = self._load_audio(input_path, sr=MODEL_SAMPLE_RATE, mono=False)
        inference_started = time.perf_counter()
        results = self.separator.separate(mix, pbar=False)
        inference_seconds = time.perf_counter() - inference_started
        missing = {
            self.spec.desired_stem,
            self.spec.secondary_stem,
        }.difference(results)
        if missing:
            raise RuntimeError("模型缺少输出 stem: %s" % ", ".join(sorted(missing)))
        for stem in (self.spec.desired_stem, self.spec.secondary_stem):
            if not np.isfinite(np.asarray(results[stem])).all():
                raise FloatingPointError("模型输出包含 NaN/Inf: %s" % stem)

        file_stem = Path(input_path).stem
        encode_started = time.perf_counter()
        futures = (
            self._save_pool.submit(
                self._save_output,
                results[self.spec.desired_stem],
                sample_rate,
                self.desired_root,
                file_stem,
                self.spec.desired_suffix,
            ),
            self._save_pool.submit(
                self._save_output,
                results[self.spec.secondary_stem],
                sample_rate,
                self.secondary_root,
                file_stem,
                self.spec.secondary_suffix,
            ),
        )
        wait(futures)
        outputs = [future.result() for future in futures]
        encode_seconds = time.perf_counter() - encode_started
        del results, mix
        return {
            "outputs": [path for path, _ in outputs],
            "inference_seconds": inference_seconds,
            "encode_seconds": encode_seconds,
        }

    def close(self):
        save_pool, self._save_pool = getattr(self, "_save_pool", None), None
        separator, self.separator = getattr(self, "separator", None), None
        try:
            if save_pool is not None:
                try:
                    save_pool.shutdown(wait=True)
                except Exception:
                    logger.exception("Failed to shut down MSST output workers")
            if separator is not None:
                try:
                    separator.close()
                except Exception:
                    logger.exception("Failed to close MSST separator")
        finally:
            del save_pool, separator
            try:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                logger.exception("Failed to clear Torch cache after MSST task")


def _worker_emit(event):
    with PYMSS_WORKER_OUTPUT_LOCK:
        sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
        sys.stdout.flush()


class _WorkerEventLogHandler(logging.Handler):
    def emit(self, record):
        try:
            _worker_emit(
                {
                    "event": "log",
                    "level": record.levelname,
                    "message": self.format(record),
                }
            )
        except Exception:
            self.handleError(record)


def _looks_like_dml_oom(error):
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "out of memory",
            "e_outofmemory",
            "not enough memory",
            "failed to allocate",
            "allocation failed",
        )
    )


def _fp16_retryable(error, requested_dtype, allow_fp32_retry):
    return (
        requested_dtype == "float16"
        and allow_fp32_retry
        and isinstance(
            error,
            (RuntimeError, NotImplementedError, FloatingPointError, TypeError, ValueError),
        )
        and not _looks_like_dml_oom(error)
    )


def _pymss_worker_main(request_path):
    request = {}
    use_dml = False
    requested_dtype = "auto"
    allow_fp32_retry = False
    try:
        with open(request_path, "r", encoding="utf-8") as request_file:
            request = json.load(request_file)
        device_type = torch.device(config.device).type
        use_dml = device_type == "privateuseone"

        spec = resolve_model(request["model_id"])
        input_paths = [os.path.abspath(path) for path in request["input_paths"]]
        if not input_paths:
            raise ValueError("没有找到可处理的音频文件")
        requested_dtype = (
            _normalize_dml_model_dtype(request.get("model_dtype", "float32"))
            if use_dml
            else "auto"
        )
        allow_fp32_retry = use_dml and bool(
            request.get("allow_fp32_retry", False)
        )
        successful_files = 0
        failed_files = 0
        file_count = len(input_paths)
        progress_context = {"file_index": 0, "path": ""}
        progress_emit_state = {"time": 0.0, "message": None, "total": None}

        def emit_progress(done, total, message):
            done = max(0.0, float(done or 0))
            total = max(1.0, float(total or 1))
            done = min(done, total)
            message = str(message or "正在处理音频")
            now = time.monotonic()
            is_edge = done <= 0 or done >= total
            context_changed = (
                progress_emit_state["message"] != message
                or progress_emit_state["total"] != total
            )
            if (
                not is_edge
                and not context_changed
                and now - progress_emit_state["time"] < 0.1
            ):
                return
            progress_emit_state.update(
                {"time": now, "message": message, "total": total}
            )
            _worker_emit(
                {
                    "event": "progress",
                    "file_index": progress_context["file_index"],
                    "file_count": file_count,
                    "path": progress_context["path"],
                    "done": done,
                    "total": total,
                    "message": message,
                }
            )

        worker_logger = logging.getLogger("rvc.pymss.worker")
        worker_logger.handlers.clear()
        worker_logger.setLevel(logging.INFO)
        worker_logger.propagate = False
        worker_log_handler = _WorkerEventLogHandler()
        worker_log_handler.setLevel(logging.INFO)
        worker_log_handler.setFormatter(logging.Formatter("%(message)s"))
        worker_logger.addHandler(worker_log_handler)

        with MSSTBatchSeparator(
            spec,
            request["output_format"],
            request["desired_root"],
            request["secondary_root"],
            model_dtype=requested_dtype,
            progress_callback=emit_progress,
            separation_logger=worker_logger,
        ) as batch:
            device_label = "DirectML" if use_dml else device_type.upper()
            _worker_emit(
                {
                    "event": "status",
                    "message": "%s 模型已加载 | 参数精度 %s"
                    % (device_label, batch.model_dtype.upper()),
                    "model_dtype": batch.model_dtype,
                }
            )
            for file_index, input_path in enumerate(input_paths, 1):
                progress_context.update(
                    {"file_index": file_index, "path": input_path}
                )
                progress_emit_state.update(
                    {"time": 0.0, "message": None, "total": None}
                )
                _worker_emit(
                    {
                        "event": "file_start",
                        "file_index": file_index,
                        "file_count": file_count,
                        "path": input_path,
                        "message": "[%s/%s] 开始处理 %s"
                        % (file_index, file_count, os.path.basename(input_path)),
                    }
                )
                try:
                    result = batch.separate_file(input_path)
                    if use_dml:
                        message = "%s -> 成功 | %s | 推理 %.2fs | 编码 %.2fs" % (
                            os.path.basename(input_path),
                            batch.model_dtype.upper(),
                            result["inference_seconds"],
                            result["encode_seconds"],
                        )
                    else:
                        message = "%s -> 成功 | 推理 %.2fs | 编码 %.2fs" % (
                            os.path.basename(input_path),
                            result["inference_seconds"],
                            result["encode_seconds"],
                        )
                    successful_files += 1
                    _worker_emit(
                        {
                            "event": "file",
                            "ok": True,
                            "file_index": file_index,
                            "file_count": file_count,
                            "path": input_path,
                            "message": message,
                        }
                    )
                except Exception as error:
                    if (
                        use_dml
                        and successful_files == 0
                        and _fp16_retryable(
                            error, requested_dtype, allow_fp32_retry
                        )
                    ):
                        _worker_emit(
                            {
                                "event": "retry_fp32",
                                "message": "DirectML FP16 路径不兼容，准备改用 FP32",
                                "detail": traceback.format_exc(),
                                "file_index": file_index,
                                "file_count": file_count,
                            }
                        )
                        return 75
                    failed_files += 1
                    _worker_emit(
                        {
                            "event": "file",
                            "ok": False,
                            "file_index": file_index,
                            "file_count": file_count,
                            "path": input_path,
                            "message": "%s -> 失败\n%s"
                            % (os.path.basename(input_path), traceback.format_exc()),
                        }
                    )

        _worker_emit(
            {
                "event": "done",
                "processed": len(input_paths),
                "file_count": file_count,
                "successful": successful_files,
                "failed": failed_files,
                "model_dtype": requested_dtype,
            }
        )
        return 0
    except BaseException as error:
        try:
            retry_fp32 = use_dml and _fp16_retryable(
                error, requested_dtype, allow_fp32_retry
            )
            _worker_emit(
                {
                    "event": "fatal",
                    "message": "%s子进程失败\n%s"
                    % ("DML " if use_dml else "PyMSS ", traceback.format_exc()),
                    "retry_fp32": retry_fp32,
                }
            )
        except Exception:
            pass
        return 1


def _pymss_dml_worker_main(request_path):
    return _pymss_worker_main(request_path)


def _begin_pymss_task():
    with PYMSS_WORKER_STATE_LOCK:
        if PYMSS_WORKER_STATE["active"]:
            return None
        PYMSS_WORKER_STATE["task_id"] += 1
        PYMSS_WORKER_STATE["active"] = True
        PYMSS_WORKER_STATE["process"] = None
        PYMSS_WORKER_STATE["stop_requested"] = False
        return PYMSS_WORKER_STATE["task_id"]


def _finish_pymss_task(task_id):
    with PYMSS_WORKER_STATE_LOCK:
        if PYMSS_WORKER_STATE["task_id"] != task_id:
            return
        PYMSS_WORKER_STATE["active"] = False
        PYMSS_WORKER_STATE["process"] = None
        PYMSS_WORKER_STATE["stop_requested"] = False


def _pymss_task_stop_requested(task_id):
    with PYMSS_WORKER_STATE_LOCK:
        return (
            PYMSS_WORKER_STATE["task_id"] == task_id
            and PYMSS_WORKER_STATE["stop_requested"]
        )


def _register_worker(task_id, process):
    with PYMSS_WORKER_STATE_LOCK:
        if (
            PYMSS_WORKER_STATE["task_id"] != task_id
            or not PYMSS_WORKER_STATE["active"]
        ):
            return True
        PYMSS_WORKER_STATE["process"] = process
        return PYMSS_WORKER_STATE["stop_requested"]


def _unregister_worker(task_id, process):
    with PYMSS_WORKER_STATE_LOCK:
        if (
            PYMSS_WORKER_STATE["task_id"] == task_id
            and PYMSS_WORKER_STATE["process"] is process
        ):
            PYMSS_WORKER_STATE["process"] = None


def stop_pymss_separation():
    with PYMSS_WORKER_STATE_LOCK:
        if not PYMSS_WORKER_STATE["active"]:
            return "当前没有正在运行的 PyMSS 任务。"
        PYMSS_WORKER_STATE["stop_requested"] = True
        process = PYMSS_WORKER_STATE["process"]
    if process is not None:
        kill_process_tree(process, "PyMSS", logger)
    return "已请求停止 PyMSS 分离任务。"


def _read_worker_log(path, limit=12000):
    try:
        with open(path, "rb") as log_file:
            log_file.seek(0, os.SEEK_END)
            size = log_file.tell()
            log_file.seek(max(0, size - limit), os.SEEK_SET)
            return log_file.read().decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


def _pymss_worker_events(
    task_id,
    spec,
    input_paths,
    desired_root,
    secondary_root,
    output_format,
    model_dtype,
    allow_fp32_retry,
):
    request = {
        "model_id": spec.model_id,
        "input_paths": input_paths,
        "desired_root": desired_root,
        "secondary_root": secondary_root,
        "output_format": output_format,
        "model_dtype": model_dtype,
        "allow_fp32_retry": allow_fp32_retry,
    }
    worker_code = (
        "import sys;"
        "request_path=sys.argv[1];"
        "sys.argv[:]=[sys.argv[0]];"
        "from tools.pymss_webui import _pymss_worker_main;"
        "raise SystemExit(_pymss_worker_main(request_path))"
    )

    if _pymss_task_stop_requested(task_id):
        yield {
            "event": "cancelled",
            "message": "PyMSS 分离任务已停止。",
            "file_count": len(input_paths),
        }
        return

    with tempfile.TemporaryDirectory(prefix="rvc-pymss-") as temp_root:
        request_path = Path(temp_root) / "request.json"
        stderr_path = Path(temp_root) / "worker.stderr.log"
        request_path.write_text(
            json.dumps(request, ensure_ascii=False),
            encoding="utf-8",
        )

        process = None
        return_code = None
        saw_done = False
        saw_fatal = False
        retry_event = None
        cancel_requested = False
        protocol_noise = deque(maxlen=5)
        with open(stderr_path, "wb") as stderr_file:
            try:
                environment = os.environ.copy()
                environment["PYTHONIOENCODING"] = "utf-8"
                environment["PYTHONUTF8"] = "1"
                process_group_kwargs = {}
                if os.name == "nt":
                    process_group_kwargs["creationflags"] = (
                        subprocess.CREATE_NO_WINDOW
                        | subprocess.CREATE_NEW_PROCESS_GROUP
                    )
                else:
                    process_group_kwargs["start_new_session"] = True
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-c",
                        worker_code,
                        str(request_path),
                    ],
                    cwd=str(PROJECT_ROOT),
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=stderr_file,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    **process_group_kwargs,
                )
                if _register_worker(task_id, process):
                    kill_process_tree(process, "PyMSS", logger)
                for raw_line in process.stdout:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        protocol_noise.append(line)
                        continue
                    if not isinstance(event, dict):
                        protocol_noise.append(line)
                        continue
                    event_type = event.get("event")
                    if event_type == "done":
                        saw_done = True
                        logger.info(
                            "PyMSS worker completed %s file(s)",
                            event.get("processed", 0),
                        )
                        yield event
                    elif event_type == "retry_fp32":
                        retry_event = event
                        logger.warning("PyMSS worker: %s", event.get("message", ""))
                    elif event_type == "fatal":
                        if event.get("retry_fp32"):
                            retry_event = event
                        else:
                            saw_fatal = True
                            logger.error("PyMSS worker: %s", event.get("message", ""))
                            yield event
                    elif event_type == "log":
                        log_level = getattr(
                            logging,
                            str(event.get("level", "INFO")).upper(),
                            logging.INFO,
                        )
                        logger.log(log_level, "PyMSS worker: %s", event.get("message", ""))
                        yield event
                    elif event_type == "file":
                        log_method = logger.info if event.get("ok") else logger.error
                        log_method("PyMSS worker: %s", event.get("message", ""))
                        yield event
                    elif event_type in {"file_start", "status"}:
                        logger.info("PyMSS worker: %s", event.get("message", ""))
                        yield event
                    elif event_type == "progress":
                        yield event
                    else:
                        protocol_noise.append(line)
                return_code = process.wait()
            finally:
                cancel_requested = _pymss_task_stop_requested(task_id)
                if process is not None:
                    try:
                        kill_process_tree(process, "PyMSS", logger)
                    finally:
                        try:
                            if process.stdout is not None:
                                process.stdout.close()
                        finally:
                            _unregister_worker(task_id, process)

        if protocol_noise:
            logger.warning(
                "Ignored non-protocol PyMSS worker stdout: %s",
                " | ".join(protocol_noise),
            )
        if cancel_requested:
            yield {
                "event": "cancelled",
                "message": "PyMSS 分离任务已停止。",
                "file_count": len(input_paths),
            }
            return
        if retry_event is not None:
            detail = retry_event.get("detail") or retry_event.get("message") or ""
            logger.warning("DirectML FP16 worker requested FP32 fallback: %s", detail)
            raise DMLFP16Fallback(detail)
        if saw_fatal:
            return
        if return_code != 0 or not saw_done:
            detail = _read_worker_log(stderr_path)
            if protocol_noise:
                noise = "\n".join(protocol_noise)
                detail = (
                    "%s\nstdout:\n%s" % (detail, noise)
                    if detail
                    else "stdout:\n%s" % noise
                )
            raise RuntimeError(
                "PyMSS worker exited unexpectedly (code=%s)%s"
                % (return_code, "\n%s" % detail if detail else "")
            )


def pymss_separate(
    model_name,
    inp_root,
    save_root_vocal,
    paths,
    save_root_ins,
    format0,
    event_callback=None,
):
    infos = deque(maxlen=240)
    task_id = None

    def notify(event):
        if event_callback is None:
            return
        try:
            event_callback(event)
        except Exception:
            logger.exception("PyMSS UI event callback failed")

    try:
        spec = resolve_model(model_name)
        input_paths = collect_input_paths(inp_root, paths)
        if not input_paths:
            raise ValueError("没有找到可处理的音频文件")

        task_id = _begin_pymss_task()
        if task_id is None:
            message = "已有 PyMSS 分离任务正在运行，请先停止当前任务。"
            infos.append(message)
            notify({"event": "busy", "message": message})
            yield "\n".join(infos)
            return

        device_type = torch.device(config.device).type
        use_dml = device_type == "privateuseone"
        infos.append("%s | %s | 正在加载模型" % (spec.label, spec.model_id))
        notify(
            {
                "event": "preparing",
                "message": "正在启动 PyMSS 模型子进程",
                "file_count": len(input_paths),
            }
        )
        logger.info(
            "Starting PyMSS worker: model=%s, device=%s, files=%s",
            spec.model_id,
            device_type,
            len(input_paths),
        )
        yield "\n".join(infos)

        attempts = _dml_model_dtype_attempts(spec) if use_dml else ("auto",)
        for attempt_index, model_dtype in enumerate(attempts):
            if _pymss_task_stop_requested(task_id):
                event = {
                    "event": "cancelled",
                    "message": "PyMSS 分离任务已停止。",
                    "file_count": len(input_paths),
                }
                notify(event)
                infos.append(event["message"])
                yield "\n".join(infos)
                return

            if use_dml:
                attempt_message = "DirectML 正在尝试参数精度 %s" % model_dtype.upper()
                infos.append(attempt_message)
                logger.info(attempt_message)
                notify(
                    {
                        "event": "precision_attempt",
                        "message": attempt_message,
                        "file_count": len(input_paths),
                    }
                )
                yield "\n".join(infos)

            events = _pymss_worker_events(
                task_id,
                spec,
                input_paths,
                save_root_vocal,
                save_root_ins,
                format0,
                model_dtype,
                allow_fp32_retry=(
                    use_dml
                    and model_dtype == "float16"
                    and attempt_index + 1 < len(attempts)
                ),
            )
            try:
                for event in events:
                    event_type = event.get("event")
                    notify(event)
                    if event_type == "progress":
                        if event_callback is not None:
                            yield "\n".join(infos)
                        continue
                    if event_type == "done":
                        if event_callback is not None:
                            yield "\n".join(infos)
                        continue

                    message = event.get("message")
                    if message:
                        infos.append(message)
                        yield "\n".join(infos)
                    if event_type == "cancelled":
                        return
            except DMLFP16Fallback:
                DML_FP16_DISABLED_MODEL_TYPES.add(spec.model_type)
                retry_message = (
                    "本机 DirectML 的 %s FP16 路径已降级；当前任务改用全新 FP32 子进程重试"
                    % spec.model_type
                )
                infos.append(retry_message)
                logger.warning(retry_message)
                notify(
                    {
                        "event": "retry_fp32",
                        "message": retry_message,
                        "file_count": len(input_paths),
                    }
                )
                yield "\n".join(infos)
                continue
            finally:
                events.close()
            break
    except Exception:
        if task_id is not None and _pymss_task_stop_requested(task_id):
            message = "PyMSS 分离任务已停止。"
            infos.append(message)
            notify({"event": "cancelled", "message": message})
        else:
            detail = traceback.format_exc()
            infos.append("失败\n%s" % detail)
            logger.error("PyMSS separation failed\n%s", detail)
            notify({"event": "fatal", "message": detail})
    finally:
        if task_id is not None:
            _finish_pymss_task(task_id)
    yield "\n".join(infos)
