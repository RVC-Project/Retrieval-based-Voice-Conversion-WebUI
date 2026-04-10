import math

import numpy as np
import pytest
import soundfile as sf

from tools.eval.metrics.f0_accuracy import compute_f0_rmse


def test_f0_identical(harmonic_wav):
    wav = harmonic_wav(freq=220, duration=2.0, sr=48000, filename="identical.wav")
    result = compute_f0_rmse(wav, wav, f0_method="harvest")
    assert result["value"] == pytest.approx(0.0, abs=1.0)


def test_f0_pitch_shifted(harmonic_wav):
    ref = harmonic_wav(freq=220, duration=2.0, sr=48000, filename="ref_220.wav")
    conv = harmonic_wav(freq=440, duration=2.0, sr=48000, filename="conv_440.wav")
    result = compute_f0_rmse(ref, conv, f0_method="harvest")
    assert 1100 <= result["value"] <= 1300


def test_f0_silent_audio(tmp_path):
    sr = 48000
    duration = 2.0
    silent = np.zeros(int(sr * duration), dtype=np.float32)
    ref_path = str(tmp_path / "silent_ref.wav")
    conv_path = str(tmp_path / "silent_conv.wav")
    sf.write(ref_path, silent, sr)
    sf.write(conv_path, silent, sr)

    result = compute_f0_rmse(ref_path, conv_path, f0_method="harvest")
    assert result["details"]["voiced_frame_ratio"] == 0.0
    assert math.isinf(result["value"])


def test_f0_vuv_error(harmonic_wav, tmp_path):
    ref = harmonic_wav(freq=220, duration=2.0, sr=48000, filename="ref_220.wav")

    # Create conv: first half harmonic signal, second half silence
    sr = 48000
    duration = 2.0
    n_samples = int(sr * duration)
    half = n_samples // 2
    t = np.linspace(0, duration / 2, half, endpoint=False)
    first_half = np.zeros(half, dtype=np.float64)
    for h in range(1, 9):
        first_half += (0.5 / h) * np.sin(2 * np.pi * 220 * h * t)
    second_half = np.zeros(n_samples - half)
    audio = np.concatenate([first_half, second_half]).astype(np.float32)
    conv_path = str(tmp_path / "half_silent.wav")
    sf.write(conv_path, audio, sr)

    result = compute_f0_rmse(ref, conv_path, f0_method="harvest")
    assert result["details"]["vuv_error_rate"] > 0


def test_f0_return_structure(harmonic_wav):
    wav = harmonic_wav(freq=220, duration=2.0, sr=48000, filename="structure.wav")
    result = compute_f0_rmse(wav, wav, f0_method="harvest")
    assert "value" in result
    assert "unit" in result
    assert "details" in result
    assert result["unit"] == "cents"
    assert "voiced_frame_ratio" in result["details"]
    assert "vuv_error_rate" in result["details"]
    assert "frames_total" in result["details"]
