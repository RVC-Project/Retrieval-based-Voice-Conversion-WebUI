"""Tests for Whisper CER metric — normalization, CER calculation, and Whisper integration."""

import pytest

from tools.eval.metrics.whisper_cer import normalize_japanese


# ---------------------------------------------------------------------------
# Normalization tests (no Whisper model required)
# ---------------------------------------------------------------------------


class TestNormalizeJapanese:
    def test_normalize_katakana_to_hiragana(self):
        assert normalize_japanese("アイウエオ") == "あいうえお"

    def test_normalize_fullwidth_digits(self):
        assert normalize_japanese("１２３") == "123"

    def test_normalize_punctuation_removal(self):
        assert normalize_japanese("こんにちは、世界！") == "こんにちは世界"

    def test_normalize_combined(self):
        assert normalize_japanese("アイウ１２３、。！") == "あいう123"

    def test_normalize_chooon(self):
        """長音記号の除去"""
        assert normalize_japanese("あーいーうー") == "あいう"

    def test_normalize_empty_string(self):
        assert normalize_japanese("") == ""


# ---------------------------------------------------------------------------
# CER calculation tests (jiwer direct, no Whisper model required)
# ---------------------------------------------------------------------------


class TestCERCalculation:
    def test_cer_identical_text(self):
        jiwer = pytest.importorskip("jiwer")
        cer_value = jiwer.cer("あいうえお", "あいうえお")
        assert cer_value == 0.0

    def test_cer_known_pair(self):
        jiwer = pytest.importorskip("jiwer")
        # "あいうえお" (5 chars) vs "あいうえ" (4 chars) -> 1 deletion -> CER = 1/5 = 0.2
        cer_value = jiwer.cer("あいうえお", "あいうえ")
        assert cer_value == pytest.approx(0.2, abs=1e-6)


# ---------------------------------------------------------------------------
# Whisper integration test (heavy, requires model download)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_whisper_importable():
    """Verify that whisper and jiwer are importable (placeholder for full integration test)."""
    import whisper

    jiwer = pytest.importorskip("jiwer")
    assert hasattr(whisper, "load_model")
    assert hasattr(jiwer, "cer")
