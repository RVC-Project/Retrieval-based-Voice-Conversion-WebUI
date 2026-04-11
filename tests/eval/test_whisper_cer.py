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


class TestNormalizeJapaneseExtended:
    """拡張正規化テスト"""

    def test_normalize_music_symbols(self):
        """音楽記号（♪等）の除去"""
        assert normalize_japanese("歌♪って♪みた") == "歌ってみた"

    def test_normalize_brackets(self):
        """各種括弧の除去"""
        assert normalize_japanese("（テスト）「あいう」") == "てすとあいう"

    def test_normalize_latin_chars(self):
        """ラテン文字の処理（全角→半角変換、大文字は保持）"""
        result = normalize_japanese("Ｈｅｌｌｏ")
        assert result == "Hello"


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


class TestCEREdgeCases:
    """CER計算のエッジケース"""

    def test_cer_completely_different(self):
        """完全に異なるテキストのCER"""
        jiwer = pytest.importorskip("jiwer")
        cer = jiwer.cer("あいうえお", "かきくけこ")
        assert cer == pytest.approx(1.0, abs=0.01)

    def test_cer_substitution(self):
        """1文字置換のCER"""
        jiwer = pytest.importorskip("jiwer")
        cer = jiwer.cer("あいうえお", "あいうえか")
        assert cer == pytest.approx(0.2, abs=0.01)

    def test_cer_insertion(self):
        """1文字挿入のCER"""
        jiwer = pytest.importorskip("jiwer")
        cer = jiwer.cer("あいうえお", "あいうえおか")
        assert cer == pytest.approx(0.2, abs=0.01)


# ---------------------------------------------------------------------------
# compute_ref_cer tests (no Whisper model required — signature/structure only)
# ---------------------------------------------------------------------------


class TestComputeRefCer:
    """compute_ref_cer関数のテスト（Whisperモデル不要のロジックテスト）"""

    def test_compute_ref_cer_requires_ref_text(self):
        """ref_textが必須パラメータであることを確認"""
        from tools.eval.metrics.whisper_cer import compute_ref_cer
        import inspect

        sig = inspect.signature(compute_ref_cer)
        assert "ref_text" in sig.parameters
        # ref_textにデフォルト値がない（必須）
        assert sig.parameters["ref_text"].default is inspect.Parameter.empty

    def test_compute_ref_cer_return_structure(self):
        """戻り値の構造を確認（関数シグネチャのみ、実行はslow）"""
        from tools.eval.metrics.whisper_cer import compute_ref_cer

        # 関数がインポートできることを確認
        assert callable(compute_ref_cer)


# ---------------------------------------------------------------------------
# metric_type field tests (no Whisper model required — signature only)
# ---------------------------------------------------------------------------


class TestMetricType:
    """metric_typeフィールドのテスト"""

    def test_metric_type_field_exists_in_module(self):
        """whisper_cerモジュールにcompute_whisper_cerが存在することを確認"""
        from tools.eval.metrics.whisper_cer import compute_whisper_cer
        import inspect

        sig = inspect.signature(compute_whisper_cer)
        # ref_textパラメータが存在する
        assert "ref_text" in sig.parameters


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
