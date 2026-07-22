import gc
import logging
import os
import subprocess
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from configs.config import Config


logger = logging.getLogger(__name__)
config = Config()
weight_pymss_root = Path(os.getenv("weight_pymss_root", "assets/pymss_weights"))

MODEL_SAMPLE_RATE = 44100
FFMPEG_PATH = Path(__file__).resolve().parents[2] / "ffmpeg.exe"
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
UVR_INFERENCE_LOCK = threading.Lock()


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
    def __init__(self, spec, output_format, desired_root, secondary_root):
        try:
            from pymss import MSSeparator, load_audio
        except ImportError as error:
            raise RuntimeError(
                "缺少 pymss 运行库，请安装对应的 CUDA 版 Python 3.12 requirements"
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
        device_id = parsed_device.index if use_cuda and parsed_device.index is not None else 0
        self._load_audio = load_audio
        self.model_load_count = 0
        self.separator = MSSeparator(
            model_type=spec.model_type,
            model_path=str(model_path),
            config_path=str(config_path),
            device="cuda" if use_cuda else "cpu",
            device_ids=[device_id],
            output_format=self.output_format,
            use_tta=False,
            store_dirs={},
            audio_params=AUDIO_PARAMS,
            debug=False,
            inference_params={
                "batch_size": spec.batch_size,
                "chunk_size": 352800,
                "overlap_size": spec.overlap_size,
                "standardize": False,
                "normalize": False,
                "use_amp": bool(config.is_half and use_cuda),
                "cuda_attention_backend": "default",
            },
        )
        self.separator.config.training.use_amp = bool(config.is_half and use_cuda)
        self._save_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rvc-msst-save")
        self.model_load_count = 1
        logger.info(
            "Loaded MSST model once for batch: %s, device=%s, half=%s",
            spec.model_id,
            self.separator.device,
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
        results = self.separator.separate(mix, pbar=True)
        inference_seconds = time.perf_counter() - inference_started
        missing = {
            self.spec.desired_stem,
            self.spec.secondary_stem,
        }.difference(results)
        if missing:
            raise RuntimeError("模型缺少输出 stem: %s" % ", ".join(sorted(missing)))

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
        save_pool = getattr(self, "_save_pool", None)
        if save_pool is not None:
            save_pool.shutdown(wait=True)
            self._save_pool = None
        separator = getattr(self, "separator", None)
        try:
            if separator is not None:
                separator.close()
                self.separator = None
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def pymss_separate(model_name, inp_root, save_root_vocal, paths, save_root_ins, format0):
    infos = []
    spec = resolve_model(model_name)
    try:
        input_paths = collect_input_paths(inp_root, paths)
        if not input_paths:
            raise ValueError("没有找到可处理的音频文件")
        infos.append("%s | %s | 正在加载模型" % (spec.label, spec.model_id))
        yield "\n".join(infos)
        with UVR_INFERENCE_LOCK:
            with MSSTBatchSeparator(
                spec,
                format0,
                save_root_vocal,
                save_root_ins,
            ) as batch:
                for input_path in input_paths:
                    try:
                        result = batch.separate_file(input_path)
                        infos.append(
                            "%s -> 成功 | 推理 %.2fs | 编码 %.2fs"
                            % (
                                os.path.basename(input_path),
                                result["inference_seconds"],
                                result["encode_seconds"],
                            )
                        )
                    except Exception:
                        infos.append(
                            "%s -> 失败\n%s"
                            % (os.path.basename(input_path), traceback.format_exc())
                        )
                    yield "\n".join(infos)
    except Exception:
        infos.append("失败\n%s" % traceback.format_exc())
    yield "\n".join(infos)
