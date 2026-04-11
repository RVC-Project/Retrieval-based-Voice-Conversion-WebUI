"""Integration tests for run_eval.py — CLI entrypoint for RVC evaluation metrics."""

import io
import json
import os
from contextlib import redirect_stdout

import pytest

from tools.eval.run_eval import _judge, _worst_status, THRESHOLDS


def _run_eval(args_list):
    """Run main() capturing stdout and returning parsed JSON."""
    from tools.eval.run_eval import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        main(args_list)
    return json.loads(buf.getvalue())


# ---------------------------------------------------------------------------
# Full pipeline tests
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_full_pipeline_mcd_f0(self, sine_wav):
        """Run with --metrics mcd,f0 and verify both keys exist in output."""
        ref = sine_wav(freq=440, duration=2.0, filename="ref.wav")
        conv = sine_wav(freq=442, duration=2.0, filename="conv.wav")
        result = _run_eval(["--ref", ref, "--conv", conv, "--metrics", "mcd,f0"])
        assert "mcd" in result["metrics"]
        assert "f0_rmse" in result["metrics"]

    def test_metrics_selection_mcd_only(self, sine_wav):
        """Run with --metrics mcd only; f0_rmse and whisper_cer must be absent."""
        ref = sine_wav(freq=440, duration=2.0, filename="ref.wav")
        conv = sine_wav(freq=442, duration=2.0, filename="conv.wav")
        result = _run_eval(["--ref", ref, "--conv", conv, "--metrics", "mcd"])
        assert "mcd" in result["metrics"]
        assert "f0_rmse" not in result["metrics"]
        assert "whisper_cer" not in result["metrics"]

    def test_output_file(self, sine_wav, tmp_path):
        """Write JSON to --output file and verify it is valid JSON."""
        ref = sine_wav(freq=440, duration=2.0, filename="ref.wav")
        conv = sine_wav(freq=442, duration=2.0, filename="conv.wav")
        out_path = str(tmp_path / "result.json")
        from tools.eval.run_eval import main

        main(["--ref", ref, "--conv", conv, "--metrics", "mcd", "--output", out_path])
        assert os.path.isfile(out_path)
        with open(out_path, encoding="utf-8") as f:
            data = json.load(f)
        assert "metrics" in data
        assert "mcd" in data["metrics"]

    def test_json_schema(self, sine_wav):
        """Verify all required top-level keys exist in JSON output."""
        ref = sine_wav(freq=440, duration=2.0, filename="ref.wav")
        conv = sine_wav(freq=442, duration=2.0, filename="conv.wav")
        result = _run_eval(["--ref", ref, "--conv", conv, "--metrics", "mcd"])
        required_keys = {"version", "timestamp", "files", "metrics", "overall_status", "config"}
        assert required_keys.issubset(result.keys())


# ---------------------------------------------------------------------------
# Judgment logic tests
# ---------------------------------------------------------------------------


class TestJudge:
    @pytest.mark.parametrize("value,metric,expected", [
        # MCD boundaries (updated thresholds)
        (20.0, "mcd", "PASS"),
        (24.0, "mcd", "PASS"),    # 境界: <= pass
        (24.01, "mcd", "WARN"),
        (28.0, "mcd", "WARN"),
        (31.99, "mcd", "WARN"),
        (32.0, "mcd", "FAIL"),    # 境界: >= fail
        (35.0, "mcd", "FAIL"),
        # F0 RMSE boundaries
        (15.0, "f0_rmse", "PASS"),
        (20.0, "f0_rmse", "PASS"),   # 境界
        (20.01, "f0_rmse", "WARN"),
        (30.0, "f0_rmse", "WARN"),
        (50.0, "f0_rmse", "FAIL"),   # 境界
        (60.0, "f0_rmse", "FAIL"),
        # Whisper CER boundaries
        (0.05, "whisper_cer", "PASS"),
        (0.10, "whisper_cer", "PASS"),  # 境界
        (0.15, "whisper_cer", "WARN"),
        (0.20, "whisper_cer", "FAIL"),  # 境界
        (0.30, "whisper_cer", "FAIL"),
        # Latency boundaries
        (100.0, "latency", "PASS"),
        (200.0, "latency", "PASS"),   # 境界
        (200.01, "latency", "WARN"),
        (350.0, "latency", "WARN"),
        (500.0, "latency", "FAIL"),   # 境界
        (600.0, "latency", "FAIL"),
        # Edge cases
        (0.0, "mcd", "PASS"),
        (float("inf"), "mcd", "FAIL"),
    ])
    def test_judge(self, value, metric, expected):
        assert _judge(value, THRESHOLDS[metric]) == expected


class TestWorstStatus:
    @pytest.mark.parametrize("statuses,expected", [
        (["PASS"], "PASS"),
        (["PASS", "WARN"], "WARN"),
        (["PASS", "FAIL"], "FAIL"),
        (["WARN", "FAIL"], "FAIL"),
        (["PASS", "PASS", "PASS"], "PASS"),
        (["FAIL", "FAIL"], "FAIL"),
    ])
    def test_worst_status(self, statuses, expected):
        assert _worst_status(statuses) == expected

    def test_invalid_status_raises(self):
        """不正なステータス文字列はKeyErrorを送出"""
        with pytest.raises(KeyError):
            _worst_status(["PASS", "INVALID"])


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestValidMetrics:
    def test_valid_metrics_includes_latency(self):
        """latencyがVALID_METRICSに含まれることを確認"""
        from tools.eval.run_eval import VALID_METRICS

        assert "latency" in VALID_METRICS


class TestVersion:
    def test_version_updated(self, sine_wav):
        """バージョンが0.2.0に更新されていることを確認"""
        ref = sine_wav(freq=440, duration=2.0, filename="ver_ref.wav")
        conv = sine_wav(freq=442, duration=2.0, filename="ver_conv.wav")
        result = _run_eval(["--ref", ref, "--conv", conv, "--metrics", "mcd"])
        assert result["version"] == "0.2.0"


class TestErrorHandling:
    def test_missing_file(self, sine_wav):
        """Non-existent file path should cause SystemExit."""
        ref = sine_wav(freq=440, duration=2.0, filename="ref.wav")
        with pytest.raises(SystemExit):
            _run_eval(["--ref", ref, "--conv", "/nonexistent/path/audio.wav", "--metrics", "mcd"])
