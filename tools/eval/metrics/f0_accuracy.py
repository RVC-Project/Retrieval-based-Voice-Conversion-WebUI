"""F0 RMSE in cents between a reference and converted audio."""

import logging
import os
import sys
import warnings

import librosa
import numpy as np
import pyworld as pw
import soundfile as sf
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

logger = logging.getLogger(__name__)

# Project root for importing RMVPE
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _extract_f0_rmvpe(audio: np.ndarray, sr: int, device: str = "cpu") -> np.ndarray | None:
    """Try to extract F0 using RMVPE. Returns None on failure."""
    try:
        if _PROJECT_ROOT not in sys.path:
            sys.path.insert(0, _PROJECT_ROOT)
        from infer.lib.rmvpe import RMVPE

        model_path = os.path.join(_PROJECT_ROOT, "assets", "rmvpe", "rmvpe.pt")
        if not os.path.isfile(model_path):
            logger.warning("RMVPE model not found at %s", model_path)
            return None

        # RMVPE expects 16kHz audio
        if sr != 16000:
            audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        else:
            audio_16k = audio

        is_half = False
        if "cuda" in device:
            import torch

            is_half = torch.cuda.is_available()

        rmvpe = RMVPE(model_path, is_half=is_half, device=device)
        f0 = rmvpe.infer_from_audio(audio_16k, thred=0.03)
        return f0.astype(np.float64)
    except Exception as exc:
        logger.warning("RMVPE extraction failed: %s", exc)
        return None


def _extract_f0_harvest(audio: np.ndarray, sr: int, f0_min: float, f0_max: float, hop_length: int) -> np.ndarray:
    """Extract F0 using pyworld.harvest."""
    audio_f64 = audio.astype(np.float64)
    frame_period = hop_length / sr * 1000.0  # ms
    f0, _ = pw.harvest(audio_f64, sr, f0_floor=f0_min, f0_ceil=f0_max, frame_period=frame_period)
    return f0


def compute_f0_rmse(
    ref_path: str,
    conv_path: str,
    sr: int = 48000,
    hop_length: int = 480,
    f0_method: str = "rmvpe",
    f0_min: float = 50,
    f0_max: float = 1100,
    device: str = "cpu",
) -> dict:
    """Compute F0 RMSE in cents between reference and converted audio.

    Extracts F0, aligns with fastdtw, computes cent-scale RMSE on
    mutually voiced frames. Also reports VUV error rate.
    """
    ref_audio, ref_sr = sf.read(ref_path, dtype="float32")
    conv_audio, conv_sr = sf.read(conv_path, dtype="float32")

    # Stereo to mono
    if ref_audio.ndim > 1:
        ref_audio = librosa.to_mono(ref_audio.T)
    if conv_audio.ndim > 1:
        conv_audio = librosa.to_mono(conv_audio.T)

    # Resample to target sr
    if ref_sr != sr:
        logger.debug("Resampling reference from %d to %d Hz", ref_sr, sr)
        ref_audio = librosa.resample(ref_audio, orig_sr=ref_sr, target_sr=sr)
    if conv_sr != sr:
        logger.debug("Resampling converted from %d to %d Hz", conv_sr, sr)
        conv_audio = librosa.resample(conv_audio, orig_sr=conv_sr, target_sr=sr)

    # F0 extraction
    f0_ref = None
    f0_conv = None

    if f0_method == "rmvpe":
        logger.debug("Attempting F0 extraction with RMVPE")
        f0_ref = _extract_f0_rmvpe(ref_audio, sr, device=device)
        f0_conv = _extract_f0_rmvpe(conv_audio, sr, device=device)
        if f0_ref is None or f0_conv is None:
            warnings.warn("RMVPE failed, falling back to harvest", stacklevel=2)
            f0_ref = None
            f0_conv = None

    if f0_ref is None or f0_conv is None:
        logger.debug("Extracting F0 with pyworld.harvest")
        f0_ref = _extract_f0_harvest(ref_audio, sr, f0_min, f0_max, hop_length)
        f0_conv = _extract_f0_harvest(conv_audio, sr, f0_min, f0_max, hop_length)

    logger.debug("F0 lengths: ref=%d, conv=%d", len(f0_ref), len(f0_conv))

    # DTW alignment on F0 sequences
    _, path = fastdtw(f0_ref.reshape(-1, 1), f0_conv.reshape(-1, 1), radius=1, dist=euclidean)
    path = np.array(path)

    f0_ref_aligned = f0_ref[path[:, 0]]
    f0_conv_aligned = f0_conv[path[:, 1]]

    frames_total = len(path)

    # Voiced/unvoiced masks
    ref_voiced = f0_ref_aligned > 0
    conv_voiced = f0_conv_aligned > 0

    # VUV error rate: voiced/unvoiced mismatch
    vuv_mismatch = np.sum(ref_voiced != conv_voiced)
    vuv_error_rate = float(vuv_mismatch / frames_total) if frames_total > 0 else 0.0

    # Both voiced frames for RMSE calculation
    both_voiced = ref_voiced & conv_voiced
    voiced_count = int(np.sum(both_voiced))
    voiced_frame_ratio = float(voiced_count / frames_total) if frames_total > 0 else 0.0

    if voiced_count == 0:
        logger.warning("No mutually voiced frames found")
        return {
            "value": float("inf"),
            "unit": "cents",
            "details": {
                "voiced_frame_ratio": 0.0,
                "vuv_error_rate": vuv_error_rate,
                "frames_total": frames_total,
            },
        }

    # Cent conversion: 1200 * log2(f0_conv / f0_ref)
    f0_r = f0_ref_aligned[both_voiced]
    f0_c = f0_conv_aligned[both_voiced]
    cent_diff = 1200.0 * np.log2(f0_c / f0_r)

    # RMSE
    rmse = float(np.sqrt(np.mean(cent_diff**2)))

    return {
        "value": rmse,
        "unit": "cents",
        "details": {
            "voiced_frame_ratio": voiced_frame_ratio,
            "vuv_error_rate": vuv_error_rate,
            "frames_total": frames_total,
        },
    }
