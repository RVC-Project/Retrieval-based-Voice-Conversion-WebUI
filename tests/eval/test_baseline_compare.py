"""Tests for baseline comparison and M1 baseline restoration tools."""

import json

import pytest

SAMPLE_BEFORE = {
    "version": "0.1.0",
    "timestamp": "2026-04-11T05:48:42.870459+00:00",
    "files": {"reference": "ref.wav", "converted": "conv_before.wav"},
    "metrics": {
        "mcd": {
            "value": 30.78,
            "unit": "dB",
            "details": {"frames_aligned": 11520},
            "status": "FAIL",
            "thresholds": {
                "pass": 6.0,
                "fail": 8.0,
                "unit": "dB",
                "lower_is_better": True,
            },
        },
        "f0_rmse": {
            "value": 38.92,
            "unit": "cents",
            "details": {
                "voiced_frame_ratio": 0.644,
                "vuv_error_rate": 0.0025,
                "frames_total": 14431,
            },
            "status": "WARN",
            "thresholds": {
                "pass": 20.0,
                "fail": 50.0,
                "unit": "cents",
                "lower_is_better": True,
            },
        },
    },
    "overall_status": "FAIL",
    "config": {
        "whisper_model": "large-v3",
        "whisper_lang": "ja",
        "sample_rate": 48000,
        "config_file": "configs/v2/48k.json",
    },
}

SAMPLE_AFTER = {
    "version": "0.1.0",
    "timestamp": "2026-04-11T07:22:11.971659+00:00",
    "files": {"reference": "ref.wav", "converted": "conv_after.wav"},
    "metrics": {
        "mcd": {
            "value": 29.97,
            "unit": "dB",
            "details": {"frames_aligned": 11526},
            "status": "FAIL",
            "thresholds": {
                "pass": 6.0,
                "fail": 8.0,
                "unit": "dB",
                "lower_is_better": True,
            },
        },
        "f0_rmse": {
            "value": 41.46,
            "unit": "cents",
            "details": {
                "voiced_frame_ratio": 0.637,
                "vuv_error_rate": 0.0028,
                "frames_total": 14369,
            },
            "status": "WARN",
            "thresholds": {
                "pass": 20.0,
                "fail": 50.0,
                "unit": "cents",
                "lower_is_better": True,
            },
        },
    },
    "overall_status": "FAIL",
    "config": {
        "whisper_model": "large-v3",
        "whisper_lang": "ja",
        "sample_rate": 48000,
        "config_file": "configs/v2/48k.json",
    },
}


class TestCompareResults:
    """compare_results関数のテスト"""

    def test_basic_comparison(self):
        """基本的な比較が正しく動作すること"""
        from tools.eval.baseline_compare import compare_results

        result = compare_results(SAMPLE_BEFORE, SAMPLE_AFTER)
        assert "metrics" in result
        assert "mcd" in result["metrics"]
        assert "f0_rmse" in result["metrics"]

    def test_delta_calculation(self):
        """deltaが正しく計算されること（after - before）"""
        from tools.eval.baseline_compare import compare_results

        result = compare_results(SAMPLE_BEFORE, SAMPLE_AFTER)
        mcd = result["metrics"]["mcd"]
        # 30.78 -> 29.97: delta = -0.81
        assert mcd["delta"] == pytest.approx(-0.81, abs=0.01)

    def test_delta_pct_calculation(self):
        """delta_pctが正しく計算されること"""
        from tools.eval.baseline_compare import compare_results

        result = compare_results(SAMPLE_BEFORE, SAMPLE_AFTER)
        mcd = result["metrics"]["mcd"]
        # (29.97 - 30.78) / 30.78 * 100 = -2.63%
        assert mcd["delta_pct"] == pytest.approx(-2.63, abs=0.1)

    def test_improved_flag_lower_is_better(self):
        """lower_is_betterメトリクスでimprovedフラグが正しいこと"""
        from tools.eval.baseline_compare import compare_results

        result = compare_results(SAMPLE_BEFORE, SAMPLE_AFTER)
        # MCD: 30.78 -> 29.97 (decreased = improved)
        assert result["metrics"]["mcd"]["improved"] is True
        # F0: 38.92 -> 41.46 (increased = degraded)
        assert result["metrics"]["f0_rmse"]["improved"] is False

    def test_missing_metric_in_after(self):
        """afterにないメトリクスはスキップされること"""
        from tools.eval.baseline_compare import compare_results

        after_partial = {
            "metrics": {
                "mcd": SAMPLE_AFTER["metrics"]["mcd"],
                # f0_rmse is missing
            }
        }
        result = compare_results(SAMPLE_BEFORE, after_partial)
        assert "mcd" in result["metrics"]

    def test_summary_counts(self):
        """summaryのカウントが正しいこと"""
        from tools.eval.baseline_compare import compare_results

        result = compare_results(SAMPLE_BEFORE, SAMPLE_AFTER)
        summary = result["summary"]
        assert summary["improved_count"] == 1  # MCD only
        assert summary["degraded_count"] == 1  # F0 RMSE
        assert summary["total_metrics"] == 2


class TestEvaluateGoNogo:
    """evaluate_gonogo関数のテスト"""

    def test_mcd_5pct_improvement_fail(self):
        """MCD 5%未満の改善はFAIL"""
        from tools.eval.baseline_compare import compare_results, evaluate_gonogo

        result = compare_results(SAMPLE_BEFORE, SAMPLE_AFTER)
        gonogo = evaluate_gonogo(result)
        # MCD改善は-2.63%で5%未満
        assert gonogo["criteria"]["mcd"]["status"] == "FAIL"

    def test_f0_10pct_improvement_fail(self):
        """F0 RMSE悪化はFAIL"""
        from tools.eval.baseline_compare import compare_results, evaluate_gonogo

        result = compare_results(SAMPLE_BEFORE, SAMPLE_AFTER)
        gonogo = evaluate_gonogo(result)
        assert gonogo["criteria"]["f0_rmse"]["status"] == "FAIL"

    def test_latency_skip_when_none(self):
        """レイテンシ結果がない場合はSKIP"""
        from tools.eval.baseline_compare import compare_results, evaluate_gonogo

        result = compare_results(SAMPLE_BEFORE, SAMPLE_AFTER)
        gonogo = evaluate_gonogo(result)
        assert gonogo["criteria"]["latency"]["status"] == "SKIP"

    def test_latency_pass(self):
        """レイテンシ200ms以下でPASS"""
        from tools.eval.baseline_compare import compare_results, evaluate_gonogo

        result = compare_results(SAMPLE_BEFORE, SAMPLE_AFTER)
        latency = {"value": 150.0, "unit": "ms"}
        gonogo = evaluate_gonogo(result, latency_result=latency)
        assert gonogo["criteria"]["latency"]["status"] == "GO"

    def test_latency_fail(self):
        """レイテンシ200ms超でFAIL"""
        from tools.eval.baseline_compare import compare_results, evaluate_gonogo

        result = compare_results(SAMPLE_BEFORE, SAMPLE_AFTER)
        latency = {"value": 300.0, "unit": "ms"}
        gonogo = evaluate_gonogo(result, latency_result=latency)
        assert gonogo["criteria"]["latency"]["status"] == "FAIL"


class TestFormatReport:
    """format_report関数のテスト"""

    def test_markdown_output(self):
        """Markdown形式の出力が生成されること"""
        from tools.eval.baseline_compare import (
            compare_results,
            evaluate_gonogo,
            format_report,
        )

        comparison = compare_results(SAMPLE_BEFORE, SAMPLE_AFTER)
        gonogo = evaluate_gonogo(comparison)
        report = format_report(comparison, gonogo)
        assert isinstance(report, str)
        assert "|" in report  # テーブルが含まれる
        assert "MCD" in report or "mcd" in report


class TestLoadEvalResult:
    """load_eval_result関数のテスト"""

    def test_load_valid_json(self, tmp_path):
        """有効なJSONファイルを読み込めること"""
        from tools.eval.baseline_compare import load_eval_result

        json_path = tmp_path / "test.json"
        json_path.write_text(json.dumps(SAMPLE_BEFORE), encoding="utf-8")
        result = load_eval_result(str(json_path))
        assert result["metrics"]["mcd"]["value"] == 30.78


class TestGeneratePreM1Config:
    """generate_pre_m1_config関数のテスト"""

    def test_generate_config(self, tmp_path):
        """Pre-M1設定ファイルが正しく生成されること"""
        from tools.eval.run_m1_baseline import generate_pre_m1_config

        # 現在の48k.jsonをソースとして使用
        source = r"C:\Users\yuta\Desktop\AIHUB\Retrieval-based-Voice-Conversion-WebUI\configs\v2\48k.json"
        output = str(tmp_path / "48k_pre_m1.json")

        generate_pre_m1_config(source_config=source, output_config=output)

        import os

        assert os.path.isfile(output)

        with open(output, encoding="utf-8") as f:
            config = json.load(f)

        # M1変更が元に戻されていること
        assert config["train"]["segment_size"] == 17280
        assert config["model"]["p_dropout"] == 0
        assert "weight_decay" not in config["train"]
        assert "c_mrstft" not in config["train"]
        assert "bf16_run" not in config["train"]


class TestCLI:
    """CLIのテスト"""

    def test_compare_cli(self, tmp_path):
        """compareサブコマンドが動作すること"""
        from tools.eval.baseline_compare import main

        before_path = tmp_path / "before.json"
        after_path = tmp_path / "after.json"
        before_path.write_text(json.dumps(SAMPLE_BEFORE), encoding="utf-8")
        after_path.write_text(json.dumps(SAMPLE_AFTER), encoding="utf-8")

        # stdoutに出力されることを確認（エラーなく実行される）
        main(["--before", str(before_path), "--after", str(after_path)])
