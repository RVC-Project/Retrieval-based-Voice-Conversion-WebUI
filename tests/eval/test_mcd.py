import numpy as np
import pytest
import soundfile as sf

from tools.eval.metrics.mcd import compute_mcd


def test_mcd_identical(sine_wav):
    wav = sine_wav(freq=440, duration=2.0, sr=48000, filename="identical.wav")
    result = compute_mcd(wav, wav)
    assert result["value"] < 0.01


def test_mcd_known_value(sine_wav):
    ref = sine_wav(freq=440, duration=2.0, sr=48000, filename="ref_440.wav")
    conv = sine_wav(freq=880, duration=2.0, sr=48000, filename="conv_880.wav")
    result = compute_mcd(ref, conv)
    assert result["value"] > 0


def test_mcd_different_sr(sine_wav, tmp_path):
    ref = sine_wav(freq=440, duration=2.0, sr=48000, filename="ref_48k.wav")

    # Create a 44100 Hz version of the same sine wave
    sr_conv = 44100
    t = np.linspace(0, 2.0, int(sr_conv * 2.0), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    conv_path = str(tmp_path / "conv_44k.wav")
    sf.write(conv_path, audio, sr_conv)

    result = compute_mcd(ref, conv_path)
    assert result["value"] >= 0


def test_mcd_short_audio(sine_wav):
    ref = sine_wav(freq=440, duration=0.5, sr=48000, filename="short_ref.wav")
    conv = sine_wav(freq=440, duration=0.5, sr=48000, filename="short_conv.wav")
    result = compute_mcd(ref, conv)
    assert result["value"] < 0.01


def test_mcd_stereo_mono(sine_wav, tmp_path):
    # Create a stereo reference
    sr = 48000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    mono = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    stereo = np.stack([mono, mono], axis=-1)
    stereo_path = str(tmp_path / "stereo_ref.wav")
    sf.write(stereo_path, stereo, sr)

    conv = sine_wav(freq=440, duration=2.0, sr=48000, filename="mono_conv.wav")
    result = compute_mcd(stereo_path, conv)
    assert result["value"] >= 0


def test_mcd_symmetry(sine_wav):
    """MCD(a, b) == MCD(b, a) — 対称性"""
    ref = sine_wav(freq=440, duration=2.0, sr=48000, filename="sym_ref.wav")
    conv = sine_wav(freq=880, duration=2.0, sr=48000, filename="sym_conv.wav")
    result_ab = compute_mcd(ref, conv)
    result_ba = compute_mcd(conv, ref)
    assert result_ab["value"] == pytest.approx(result_ba["value"], rel=1e-3)


def test_mcd_non_negative(sine_wav):
    """MCD >= 0 — 非負性"""
    ref = sine_wav(freq=440, duration=2.0, sr=48000, filename="nn_ref.wav")
    conv = sine_wav(freq=660, duration=2.0, sr=48000, filename="nn_conv.wav")
    result = compute_mcd(ref, conv)
    assert result["value"] >= 0


class TestMCDScale:
    """dBスケール修正に関連するテスト"""

    def test_mcd_value_range(self, sine_wav):
        """MCD値が妥当な範囲内にあることを確認（dBスケール）"""
        ref = sine_wav(freq=440, duration=2.0, sr=48000, filename="range_ref.wav")
        conv = sine_wav(freq=880, duration=2.0, sr=48000, filename="range_conv.wav")
        result = compute_mcd(ref, conv)
        # dBスケールMCDは正の有限値であること
        assert 0 < result["value"] < 500

    def test_mcd_returns_db_unit(self, sine_wav):
        """unitフィールドが'dB'であることを確認"""
        wav = sine_wav(freq=440, duration=2.0, sr=48000, filename="unit_test.wav")
        result = compute_mcd(wav, wav)
        assert result["unit"] == "dB"

    def test_mcd_frames_aligned_positive(self, sine_wav):
        """DTWアラインされたフレーム数が正であることを確認"""
        ref = sine_wav(freq=440, duration=2.0, sr=48000, filename="frames_ref.wav")
        conv = sine_wav(freq=880, duration=2.0, sr=48000, filename="frames_conv.wav")
        result = compute_mcd(ref, conv)
        assert result["details"]["frames_aligned"] > 0

    def test_mcd_custom_params(self, sine_wav):
        """カスタムパラメータ（n_mels, n_mfcc）で計算できることを確認"""
        ref = sine_wav(freq=440, duration=2.0, sr=48000, filename="custom_ref.wav")
        conv = sine_wav(freq=880, duration=2.0, sr=48000, filename="custom_conv.wav")
        result_default = compute_mcd(ref, conv)
        result_custom = compute_mcd(ref, conv, n_mels=80, n_mfcc=20)
        # 異なるパラメータでも正の値を返す
        assert result_custom["value"] > 0
        # パラメータが異なれば値も変わる
        assert result_default["value"] != pytest.approx(result_custom["value"], rel=0.01)

    def test_mcd_different_lengths(self, sine_wav):
        """異なる長さの音声でもDTWで対応できることを確認"""
        ref = sine_wav(freq=440, duration=2.0, sr=48000, filename="long_ref.wav")
        conv = sine_wav(freq=440, duration=1.5, sr=48000, filename="short_conv.wav")
        result = compute_mcd(ref, conv)
        assert result["value"] >= 0
