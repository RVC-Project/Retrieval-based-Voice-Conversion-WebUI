from __future__ import annotations

import base64
import io
import json
import os
import re
import tempfile
import time
import uuid
import zipfile

import numpy as np

from ..audio_io import save_audio
from .errors import APIError


PCM_FORMATS = {
    "pcm_f32le": (np.dtype("<f4"), 4),
    "pcm_s16le": (np.dtype("<i2"), 2),
}
OUTPUT_FORMATS = {"pcm_f32le", "wav", "flac"}
RESPONSE_FORMATS = {"json", "zip"}


def parse_int(value, param, code="invalid_request"):
    """Parse an integer request parameter and raise an APIError on failure.

    Args:
        value (Any): Value value.
        param (str | None): Param value.
        code (str, optional): Code value. Defaults to 'invalid_request'.

    Returns:
        Any: Parsed value."""
    try:
        return int(value)
    except (TypeError, ValueError):
        raise APIError(400, code, f"Invalid integer for {param}.", param=param)


def normalize_stems(value, valid_instruments):
    """Normalize requested stem names against the loaded model instruments.

    Args:
        value (Any): Value value.
        valid_instruments (Any): Valid instruments value.

    Returns:
        Any: Computed result."""
    if value is None:
        return None
    if isinstance(value, str):
        raw_stems = value.split(",")
    elif isinstance(value, (list, tuple)):
        raw_stems = value
    else:
        raise APIError(400, "invalid_stem", "stems must be a string or array.", param="stems")

    lower_map = {stem.lower(): stem for stem in valid_instruments}
    selected = []
    seen = set()
    for raw in raw_stems:
        stem = str(raw).strip()
        if not stem:
            continue
        canonical = lower_map.get(stem.lower())
        if canonical is None:
            raise APIError(400, "invalid_stem", f"Invalid stem {stem!r}. Valid stems: {list(valid_instruments)}", param="stems")
        if canonical.lower() in seen:
            continue
        seen.add(canonical.lower())
        selected.append(canonical)
    return selected or None


def validate_common_options(response_format, output_audio_format):
    """Validate response and output audio format options.

    Args:
        response_format (str): Response format value.
        output_audio_format (str): Output audio format value.

    Returns:
        None: This callable completes for its side effects."""
    if response_format not in RESPONSE_FORMATS:
        raise APIError(
            400,
            "invalid_response_format",
            f"Unsupported response_format {response_format!r}.",
            param="response_format",
        )
    if output_audio_format not in OUTPUT_FORMATS:
        raise APIError(
            400,
            "invalid_output_audio_format",
            f"Unsupported output_audio_format {output_audio_format!r}.",
            param="output_audio_format",
        )
    if response_format == "json" and output_audio_format != "pcm_f32le":
        raise APIError(
            400,
            "invalid_output_audio_format",
            "response_format='json' only supports output_audio_format='pcm_f32le'.",
            param="output_audio_format",
        )


def decode_pcm(raw, audio_format, sample_rate, channels, expected_sample_rate, max_audio_seconds):
    """Decode raw PCM request bytes into a channel-first float32 array.

    Args:
        raw (bytes): Raw value.
        audio_format (Any): Audio format value.
        sample_rate (int): Audio sample rate in Hz.
        channels (int): Channels value.
        expected_sample_rate (Any): Expected sample rate value.
        max_audio_seconds (Any): Max audio seconds value.

    Returns:
        Any: Computed result."""
    if audio_format not in PCM_FORMATS:
        raise APIError(400, "invalid_audio_format", f"Unsupported audio format {audio_format!r}.", param="format")
    if channels not in (1, 2):
        raise APIError(400, "invalid_channel_count", "channels must be 1 or 2.", param="channels")
    if sample_rate != expected_sample_rate:
        raise APIError(
            400,
            "invalid_sample_rate",
            f"sample_rate must be {expected_sample_rate}.",
            param="sample_rate",
        )
    if not raw:
        raise APIError(400, "empty_audio", "Decoded PCM audio is empty.", param="input.data")

    dtype, sample_width = PCM_FORMATS[audio_format]
    frame_width = channels * sample_width
    if len(raw) % frame_width:
        raise APIError(
            400,
            "invalid_audio_length",
            "PCM bytes length is not aligned to format and channels.",
            param="input.data",
        )

    frames = len(raw) // frame_width
    seconds = frames / float(sample_rate)
    if max_audio_seconds and seconds > max_audio_seconds:
        raise APIError(413, "request_too_large", f"Audio duration exceeds {max_audio_seconds} seconds.")

    data = np.frombuffer(raw, dtype=dtype)
    if audio_format == "pcm_f32le":
        if not np.isfinite(data).all():
            raise APIError(400, "invalid_audio_data", "pcm_f32le audio contains NaN or Inf.", param="input.data")
        float_data = data.astype(np.float32, copy=False)
    else:
        float_data = data.astype(np.float32) / 32768.0

    frame_data = float_data.reshape(-1, channels)
    mix = frame_data[:, 0] if channels == 1 else frame_data.T
    return np.ascontiguousarray(mix), seconds


def audio_to_interleaved_f32(audio):
    """Convert channel-first audio to interleaved float32 samples.

    Args:
        audio (np.ndarray): Audio samples.

    Returns:
        Any: Computed result."""
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        normalized = np.ascontiguousarray(array)
        return normalized, 1
    if array.ndim != 2:
        raise ValueError(f"Expected mono or stereo audio, got shape {array.shape}")
    if array.shape[1] in (1, 2):
        normalized = np.ascontiguousarray(array)
        return normalized, int(array.shape[1])
    if array.shape[0] in (1, 2):
        normalized = np.ascontiguousarray(array.T)
        return normalized, int(array.shape[0])
    raise ValueError(f"Expected mono or stereo audio, got shape {array.shape}")


def f32le_bytes(audio):
    """Serialize audio as little-endian float32 PCM bytes.

    Args:
        audio (np.ndarray): Audio samples.

    Returns:
        Any: Computed result."""
    array, channels = audio_to_interleaved_f32(audio)
    return np.asarray(array, dtype="<f4").tobytes(), channels


def _safe_slug(value):
    """Implement the safe slug helper.

    Args:
        value (Any): Value value.

    Returns:
        Any: Computed result."""
    slug = re.sub(r"[^a-z0-9_-]+", "-", str(value).lower()).strip("-._")
    return slug or "stem"


def _filename(index, stem, output_audio_format):
    """Implement the filename helper.

    Args:
        index (Any): Index value.
        stem (str): Stem value.
        output_audio_format (str): Output audio format value.

    Returns:
        Any: Computed result."""
    extension = "f32le" if output_audio_format == "pcm_f32le" else output_audio_format
    return f"{index:04d}-{_safe_slug(stem)}.{extension}"


def _encode_container(audio, sample_rate, output_format, audio_params):
    """Encode container.

    Args:
        audio (np.ndarray): Audio samples.
        sample_rate (int): Audio sample rate in Hz.
        output_format (str | None): Output format such as wav, flac, mp3, or m4a.
        audio_params (dict | None): Encoding options for the output audio format.

    Returns:
        Any: Computed result."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, f"audio.{output_format}")
        save_audio(path, audio, sample_rate, output_format, audio_params)
        with open(path, "rb") as f:
            return f.read()


def ordered_results(results, stems, instruments):
    """Return separation results in requested stem order.

    Args:
        results (dict): Results value.
        stems (Sequence[str] | None): Requested output stem names.
        instruments (Sequence[str] | None): Instruments value.

    Returns:
        Any: Computed result."""
    order = stems or list(instruments)
    missing = [stem for stem in order if stem not in results]
    if missing:
        raise APIError(
            500,
            "separation_failed",
            f"Separation did not return stem(s): {missing}",
            error_type="server_error",
        )
    return [(stem, results[stem]) for stem in order]


def json_response(loaded, model, results, stems, input_seconds):
    """Build a JSON separation response with base64 audio payloads.

    Args:
        loaded (LoadedModel): Loaded value.
        model (str): Model value.
        results (dict): Results value.
        stems (Sequence[str] | None): Requested output stem names.
        input_seconds (Any): Input seconds value.

    Returns:
        Any: Computed result."""
    outputs = []
    for stem, audio in ordered_results(results, stems, loaded.instruments):
        raw, channels = f32le_bytes(audio)
        outputs.append(
            {
                "stem": stem,
                "audio": {
                    "format": "pcm_f32le",
                    "sample_rate": loaded.sample_rate,
                    "channels": channels,
                    "data": base64.b64encode(raw).decode("ascii"),
                },
            }
        )
    created = int(time.time())
    return {
        "id": "sep_" + uuid.uuid4().hex,
        "object": "audio.separation",
        "created": created,
        "model": model,
        "outputs": outputs,
        "metadata": {
            "input_seconds": input_seconds,
            "output_stems": [item["stem"] for item in outputs],
            "device": loaded.device,
        },
        "usage": {
            "type": "duration",
            "seconds": input_seconds,
        },
    }


def zip_response(loaded, model, results, stems, input_seconds, output_audio_format):
    """Build a ZIP separation response with encoded audio files.

    Args:
        loaded (LoadedModel): Loaded value.
        model (str): Model value.
        results (dict): Results value.
        stems (Sequence[str] | None): Requested output stem names.
        input_seconds (Any): Input seconds value.
        output_audio_format (str): Output audio format value.

    Returns:
        Any: Computed result."""
    output_items = []
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for index, (stem, audio) in enumerate(ordered_results(results, stems, loaded.instruments), start=1):
            filename = _filename(index, stem, output_audio_format)
            if output_audio_format == "pcm_f32le":
                content, channels = f32le_bytes(audio)
                format_name = "pcm_f32le"
            else:
                content = _encode_container(
                    audio,
                    loaded.sample_rate,
                    output_audio_format,
                    loaded.audio_params,
                )
                _, channels = audio_to_interleaved_f32(audio)
                format_name = output_audio_format
            zf.writestr(filename, content)
            output_items.append(
                {
                    "stem": stem,
                    "filename": filename,
                    "format": format_name,
                    "sample_rate": loaded.sample_rate,
                    "channels": channels,
                }
            )

        manifest = {
            "id": "sep_" + uuid.uuid4().hex,
            "object": "audio.separation",
            "model": model,
            "outputs": output_items,
            "usage": {
                "type": "duration",
                "seconds": input_seconds,
            },
        }
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return archive.getvalue()
