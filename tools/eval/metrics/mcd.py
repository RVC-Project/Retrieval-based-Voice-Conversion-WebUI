"""Mel Cepstral Distortion (MCD) between a reference and converted audio."""

import logging
import math
import os
import sys

import librosa
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tools.eval.audio_utils import load_audio

logger = logging.getLogger(__name__)


def compute_mcd(
    ref_path: str,
    conv_path: str,
    sr: int = 48000,
    n_mels: int = 40,
    n_mfcc: int = 13,
    fmin: float = 0.0,
    fmax: float | None = None,
    hop_length: int = 480,
) -> dict:
    """Compute Mel Cepstral Distortion between reference and converted audio.

    Uses MFCC coefficients 1-12 (excluding 0th) with fastdtw alignment.
    Returns MCD in dB.
    """
    ref_audio = load_audio(ref_path, target_sr=sr)
    conv_audio = load_audio(conv_path, target_sr=sr)

    # Extract MFCC (use coefficients 1 through n_mfcc-1, excluding 0th)
    mfcc_ref = librosa.feature.mfcc(
        y=ref_audio, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels,
        fmin=fmin, fmax=fmax, hop_length=hop_length,
    )
    mfcc_conv = librosa.feature.mfcc(
        y=conv_audio, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels,
        fmin=fmin, fmax=fmax, hop_length=hop_length,
    )

    # Remove 0th coefficient -> shape (n_mfcc-1, T)
    mfcc_ref = mfcc_ref[1:]
    mfcc_conv = mfcc_conv[1:]

    # Transpose to (T, n_mfcc-1) for DTW
    mfcc_ref = mfcc_ref.T
    mfcc_conv = mfcc_conv.T

    logger.debug("MFCC shapes: ref=%s, conv=%s", mfcc_ref.shape, mfcc_conv.shape)

    # DTW alignment
    _, path = fastdtw(mfcc_ref, mfcc_conv, radius=20, dist=euclidean)
    path = np.array(path)

    ref_aligned = mfcc_ref[path[:, 0]]
    conv_aligned = mfcc_conv[path[:, 1]]

    frames_aligned = len(path)
    logger.debug("DTW aligned frames: %d", frames_aligned)

    # MCD = (10 * sqrt(2) / ln(10)) * mean(||mfcc_ref - mfcc_conv||_2)
    diff = ref_aligned - conv_aligned
    frame_dists = np.sqrt(np.sum(diff**2, axis=1))
    coeff = 10.0 * math.sqrt(2.0) / math.log(10.0)
    mcd_value = float(coeff * np.mean(frame_dists))

    return {
        "value": mcd_value,
        "unit": "dB",
        "details": {
            "frames_aligned": frames_aligned,
        },
    }
