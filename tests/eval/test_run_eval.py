"""Integration tests for run_eval.py — CLI entrypoint for RVC evaluation metrics."""

import io
import json
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from contextlib import redirect_stdout

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
    def test_mcd_pass(self):
        assert _judge(5.0, THRESHOLDS["mcd"]) == "PASS"

    def test_mcd_warn(self):
        assert _judge(7.0, THRESHOLDS["mcd"]) == "WARN"

    def test_mcd_fail(self):
        assert _judge(9.0, THRESHOLDS["mcd"]) == "FAIL"

    def test_f0_pass(self):
        assert _judge(15.0, THRESHOLDS["f0_rmse"]) == "PASS"

    def test_f0_warn(self):
        assert _judge(30.0, THRESHOLDS["f0_rmse"]) == "WARN"

    def test_f0_fail(self):
        assert _judge(60.0, THRESHOLDS["f0_rmse"]) == "FAIL"


class TestWorstStatus:
    def test_pass_warn(self):
        assert _worst_status(["PASS", "WARN"]) == "WARN"

    def test_pass_fail(self):
        assert _worst_status(["PASS", "FAIL"]) == "FAIL"

    def test_all_pass(self):
        assert _worst_status(["PASS", "PASS", "PASS"]) == "PASS"

    def test_warn_fail(self):
        assert _worst_status(["WARN", "FAIL"]) == "FAIL"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_missing_file(self, sine_wav):
        """Non-existent file path should cause SystemExit."""
        ref = sine_wav(freq=440, duration=2.0, filename="ref.wav")
        with pytest.raises(SystemExit):
            _run_eval(["--ref", ref, "--conv", "/nonexistent/path/audio.wav", "--metrics", "mcd"])
