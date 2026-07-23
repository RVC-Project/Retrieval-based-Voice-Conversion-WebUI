from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from .audio_io import load_audio, save_audio

ENSEMBLE_ALGORITHMS = (
    "avg_wave",
    "median_wave",
    "min_wave",
    "max_wave",
    "avg_fft",
    "median_fft",
    "min_fft",
    "max_fft",
)


def _as_channel_first(audio):
    """Implement the as channel first helper.

    Args:
        audio (np.ndarray): Audio samples.

    Returns:
        Any: Computed result."""
    audio = np.asarray(audio, dtype=np.float32)
    return audio[None, :] if audio.ndim == 1 else audio


def stft(wave, nfft=2048, hl=1024):
    """Implement the stft helper.

    Args:
        wave (np.ndarray): Wave value.
        nfft (Any, optional): Nfft value. Defaults to 2048.
        hl (Any, optional): Hl value. Defaults to 1024.

    Returns:
        Any: Computed result."""
    wave = _as_channel_first(wave)
    return np.asfortranarray([librosa.stft(np.asfortranarray(channel), n_fft=nfft, hop_length=hl) for channel in wave])


def istft(spec, hl=1024, length=None):
    """Implement the istft helper.

    Args:
        spec (np.ndarray): Spec value.
        hl (Any, optional): Hl value. Defaults to 1024.
        length (Any, optional): Length value. Defaults to None.

    Returns:
        Any: Computed result."""
    return np.asfortranarray([librosa.istft(np.asfortranarray(channel), hop_length=hl, length=length) for channel in spec])


def absmax(a, *, axis):
    """Implement the absmax helper.

    Args:
        a (np.ndarray): A value.
        axis (Any): Axis value.

    Returns:
        Any: Computed result."""
    dims = list(a.shape)
    dims.pop(axis)
    indices = np.ogrid[tuple(slice(0, d) for d in dims)]
    argmax = np.abs(a).argmax(axis=axis)
    indices.insert((len(a.shape) + axis) % len(a.shape), argmax)
    return a[tuple(indices)]


def lambda_min(arr, axis=None, key=None, keepdims=False):
    """Implement the lambda min helper.

    Args:
        arr (np.ndarray): Arr value.
        axis (Any, optional): Axis value. Defaults to None.
        key (str, optional): Key value. Defaults to None.
        keepdims (Any, optional): Keepdims value. Defaults to False.

    Returns:
        Any: Computed result."""
    idxs = np.argmin(key(arr), axis)
    if axis is None:
        return arr.flatten()[idxs]
    idxs = np.expand_dims(idxs, axis)
    result = np.take_along_axis(arr, idxs, axis)
    return result if keepdims else np.squeeze(result, axis=axis)


def lambda_max(arr, axis=None, key=None, keepdims=False):
    """Implement the lambda max helper.

    Args:
        arr (np.ndarray): Arr value.
        axis (Any, optional): Axis value. Defaults to None.
        key (str, optional): Key value. Defaults to None.
        keepdims (Any, optional): Keepdims value. Defaults to False.

    Returns:
        Any: Computed result."""
    idxs = np.argmax(key(arr), axis)
    if axis is None:
        return arr.flatten()[idxs]
    idxs = np.expand_dims(idxs, axis)
    result = np.take_along_axis(arr, idxs, axis)
    return result if keepdims else np.squeeze(result, axis=axis)


def average_waveforms(pred_track, weights=None, algorithm="avg_wave"):
    """Combine source waveforms with a selected ensemble algorithm.

    Args:
        pred_track (Any): Pred track value.
        weights (Sequence[float] | None, optional): Per-file ensemble weights. Defaults to equal weights when None. Defaults to None.
        algorithm (str, optional): Ensemble algorithm name. Defaults to "avg_wave".

    Returns:
        np.ndarray: Combined waveform shaped as channels by samples.

    Example:
        >>> combined = average_waveforms(predictions, weights=[1, 1], algorithm="avg_wave")"""
    if algorithm not in ENSEMBLE_ALGORITHMS:
        raise ValueError(f"Unknown ensemble algorithm: {algorithm}")

    pred_track = np.asarray(pred_track, dtype=np.float32)
    if pred_track.ndim != 3:
        raise ValueError("pred_track must have shape (files, channels, samples)")

    if weights is None:
        weights = np.ones(pred_track.shape[0], dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if weights.shape != (pred_track.shape[0],):
        raise ValueError("weights length must match number of input files")
    if algorithm in {"avg_wave", "avg_fft"} and np.isclose(weights.sum(), 0.0):
        raise ValueError("weights must not sum to zero for average ensemble algorithms")

    final_length = pred_track.shape[-1]
    mod_track = []
    for idx in range(pred_track.shape[0]):
        if algorithm == "avg_wave":
            mod_track.append(pred_track[idx] * weights[idx])
        elif algorithm in {"median_wave", "min_wave", "max_wave"}:
            mod_track.append(pred_track[idx])
        elif algorithm in {"avg_fft", "median_fft", "min_fft", "max_fft"}:
            spec = stft(pred_track[idx], nfft=2048, hl=1024)
            mod_track.append(spec * weights[idx] if algorithm == "avg_fft" else spec)

    pred_track = np.asarray(mod_track)
    if algorithm == "avg_wave":
        return pred_track.sum(axis=0) / weights.sum()
    if algorithm == "median_wave":
        return np.median(pred_track, axis=0)
    if algorithm == "min_wave":
        return lambda_min(pred_track, axis=0, key=np.abs)
    if algorithm == "max_wave":
        return lambda_max(pred_track, axis=0, key=np.abs)
    if algorithm == "avg_fft":
        return istft(pred_track.sum(axis=0) / weights.sum(), hl=1024, length=final_length)
    if algorithm == "min_fft":
        return istft(lambda_min(pred_track, axis=0, key=np.abs), hl=1024, length=final_length)
    if algorithm == "max_fft":
        return istft(absmax(pred_track, axis=0), hl=1024, length=final_length)
    if algorithm == "median_fft":
        return istft(np.median(pred_track, axis=0), hl=1024, length=final_length)

    raise AssertionError("unreachable")


def ensemble_audios(files, algorithm="avg_wave", weights=None, logger=None):
    """Load and combine multiple audio files with an ensemble algorithm.

    All input files must have the same sample rate and channel count. If input
    lengths differ, every file is truncated to the shortest length before
    combining. ``avg_*`` algorithms use weights; median/min/max algorithms
    ignore weights except for input validation.

    Args:
        files (Sequence[str | os.PathLike]): Audio files to combine. At least
            two files are required.
        algorithm (str, optional): Ensemble algorithm. Supported values are
            ``avg_wave``, ``median_wave``, ``min_wave``, ``max_wave``,
            ``avg_fft``, ``median_fft``, ``min_fft``, and ``max_fft``.
            Defaults to ``"avg_wave"``.
        weights (Sequence[float] | None, optional): Per-file weights. When
            ``None``, every file gets weight ``1``. For average algorithms the
            weight sum must not be zero. Defaults to None.
        logger (logging.Logger | None, optional): Optional logger used for
            debug messages and length-truncation warnings. Defaults to None.

    Returns:
        tuple[np.ndarray, int]: Combined audio shaped as samples by channels,
        and the sample rate.

    Raises:
        ValueError: If fewer than two files are provided, weights length does
            not match input count, sample rates differ, channel counts differ,
            or the algorithm is unknown.
        FileNotFoundError: If any input file does not exist.

    Example:
        >>> from pymss import ensemble_audios
        >>> audio, sample_rate = ensemble_audios(
        ...     ["vocals_a.wav", "vocals_b.wav"],
        ...     algorithm="avg_wave",
        ...     weights=[0.7, 0.3],
        ... )

    Example:
        >>> audio, sample_rate = ensemble_audios(
        ...     ["stem_a.wav", "stem_b.wav", "stem_c.wav"],
        ...     algorithm="median_fft",
        ... )"""
    if len(files) < 2:
        raise ValueError("at least two input files are required")

    if weights is None:
        weights = np.ones(len(files), dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if weights.shape != (len(files),):
        raise ValueError("weights length must match number of input files")

    data = []
    sample_rate = None
    for file in files:
        path = Path(file)
        if not path.is_file():
            raise FileNotFoundError(f"input audio file not found: {path}")
        audio, sr = load_audio(str(path), sr=None, mono=False)
        audio = _as_channel_first(audio)
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise ValueError(f"sample rate mismatch: {path} has {sr}, expected {sample_rate}")
        data.append(audio)
        if logger is not None:
            logger.debug("read %s, waveform shape=%s, sample_rate=%s", path, audio.shape, sr)

    channel_counts = {item.shape[0] for item in data}
    if len(channel_counts) != 1:
        raise ValueError("all input files must have the same channel count")

    lengths = [item.shape[-1] for item in data]
    min_length = min(lengths)
    if len(set(lengths)) > 1:
        if logger is not None:
            logger.warning("Input audio files have different lengths. Truncating all to the shortest length.")
        data = [item[..., :min_length] for item in data]

    result = average_waveforms(np.asarray(data), weights=weights, algorithm=algorithm)
    if logger is not None:
        logger.debug("ensemble result shape=%s", result.shape)
    return result.T, sample_rate


def save_ensemble_audio(
    files,
    output,
    algorithm="avg_wave",
    weights=None,
    output_format=None,
    audio_params=None,
    logger=None,
):
    """Combine audio files and save the ensemble result.

    This is the file-writing wrapper around ``ensemble_audios(...)``. The
    output format is inferred from ``output`` when it has a suffix, otherwise
    ``output_format`` is used. If neither is provided, ``.wav`` is added.

    Args:
        files (Sequence[str | os.PathLike]): Audio files to combine. At least
            two files are required.
        output (str | os.PathLike): Output file path. A missing suffix becomes
            ``.wav`` unless ``output_format`` is provided.
        algorithm (str, optional): Ensemble algorithm. Supported values are
            ``avg_wave``, ``median_wave``, ``min_wave``, ``max_wave``,
            ``avg_fft``, ``median_fft``, ``min_fft``, and ``max_fft``.
            Defaults to ``"avg_wave"``.
        weights (Sequence[float] | None, optional): Per-file weights. Defaults
            to equal weights when None.
        output_format (str | None, optional): Explicit output format such as
            ``wav``, ``flac``, ``mp3``, or ``m4a``. Defaults to None.
        audio_params (dict | None, optional): Encoding options forwarded to
            ``save_audio(...)``. Examples include
            ``{"wav_bit_depth": "FLOAT"}``,
            ``{"flac_bit_depth": "PCM_24"}``, or
            ``{"mp3_bit_rate": "320k"}``. Defaults to None.
        logger (logging.Logger | None, optional): Optional logger for progress
            messages. Defaults to None.

    Returns:
        pathlib.Path: Final output path.

    Raises:
        ValueError: If inputs cannot be ensembled.
        FileNotFoundError: If any input file does not exist.

    Example:
        >>> from pymss import save_ensemble_audio
        >>> save_ensemble_audio(
        ...     ["vocals_a.wav", "vocals_b.wav"],
        ...     "vocals_ensemble.flac",
        ...     algorithm="avg_wave",
        ...     weights=[1, 1],
        ...     audio_params={"flac_bit_depth": "PCM_24"},
        ... )

    Example:
        >>> save_ensemble_audio(["a.wav", "b.wav"], "ensemble", output_format="wav")"""
    result, sample_rate = ensemble_audios(files, algorithm=algorithm, weights=weights, logger=logger)
    output_path = Path(output)
    if not output_path.suffix and not output_format:
        output_path = output_path.with_suffix(".wav")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_format = output_format or output_path.suffix.lstrip(".").lower() or "wav"
    save_audio(str(output_path), result, sample_rate, output_format, audio_params or {})
    return output_path
