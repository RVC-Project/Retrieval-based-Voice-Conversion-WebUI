"""音声ファイルの読み込み・前処理ユーティリティ"""

import logging

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def load_audio(path: str, target_sr: int) -> np.ndarray:
    """音声ファイルを読み込み、モノラル化・リサンプルして返す。

    Args:
        path: WAVファイルパス
        target_sr: 目標サンプリングレート

    Returns:
        float32 numpy array (1D)
    """
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = librosa.to_mono(audio.T)
    if sr != target_sr:
        logger.debug("Resampling from %d to %d Hz", sr, target_sr)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio
