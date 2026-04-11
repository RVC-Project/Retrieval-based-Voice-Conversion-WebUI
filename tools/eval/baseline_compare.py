"""Baseline comparison tool for RVC evaluation results.

2つの評価結果JSON（run_eval.pyの出力）を比較し、
各メトリクスの変化量とGo/No-Go判定を行う。

Usage (CLI):
    uv run python tools/eval/baseline_compare.py \\
        --before baseline.json --after improved.json

Usage (library):
    from tools.eval.baseline_compare import compare_results, evaluate_gonogo
    comparison = compare_results(before, after)
    gonogo = evaluate_gonogo(comparison)
"""

import argparse
import json
import logging
import sys

logger = logging.getLogger(__name__)

# 全メトリクスは lower_is_better（run_eval.py の THRESHOLDS 準拠）
_LOWER_IS_BETTER = {"mcd", "f0_rmse", "whisper_cer"}

# Go/No-Go判定のデフォルト基準
DEFAULT_GONOGO_CRITERIA = {
    "mcd": {
        "metric": "mcd",
        "direction": "decrease",
        "threshold_pct": 5.0,
        "label": "MCD改善",
    },
    "f0_rmse": {
        "metric": "f0_rmse",
        "direction": "decrease",
        "threshold_pct": 10.0,
        "label": "F0 RMSE改善",
    },
    "latency": {
        "metric": "latency",
        "direction": "below",
        "threshold_abs": 200.0,
        "label": "レイテンシ200ms以下",
    },
}


def load_eval_result(json_path: str) -> dict:
    """評価結果JSONを読み込む。

    Args:
        json_path: run_eval.py が出力したJSONファイルのパス

    Returns:
        評価結果の辞書
    """
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def compare_results(before: dict, after: dict) -> dict:
    """2つの評価結果を比較し、各メトリクスの変化を計算する。

    両方に存在するメトリクスのみ比較する。片方にしか存在しないメトリクスはスキップ。

    Args:
        before: 改善前の評価結果（run_eval.py出力）
        after: 改善後の評価結果（run_eval.py出力）

    Returns:
        dict with structure::

            {
                "metrics": {
                    "mcd": {
                        "before": 30.78, "after": 29.97,
                        "delta": -0.81, "delta_pct": -2.63,
                        "improved": True, "unit": "dB"
                    },
                    ...
                },
                "summary": {
                    "improved_count": 2,
                    "degraded_count": 1,
                    "unchanged_count": 0,
                    "total_metrics": 3,
                },
                "skipped": ["whisper_cer"]
            }
    """
    before_metrics = before.get("metrics", {})
    after_metrics = after.get("metrics", {})

    all_keys = set(before_metrics.keys()) | set(after_metrics.keys())
    common_keys = set(before_metrics.keys()) & set(after_metrics.keys())
    skipped = sorted(all_keys - common_keys)

    if skipped:
        logger.info("片方にのみ存在するためスキップ: %s", skipped)

    metrics = {}
    improved_count = 0
    degraded_count = 0
    unchanged_count = 0

    for key in sorted(common_keys):
        bval = before_metrics[key]["value"]
        aval = after_metrics[key]["value"]
        unit = after_metrics[key].get("unit", "")

        delta = aval - bval
        if bval != 0:
            delta_pct = (delta / abs(bval)) * 100.0
        else:
            delta_pct = 0.0 if delta == 0 else float("inf")

        # lower_is_better のメトリクスは delta < 0 が改善
        if key in _LOWER_IS_BETTER:
            if delta < 0:
                improved = True
            elif delta > 0:
                improved = False
            else:
                improved = None  # 変化なし
        else:
            # higher_is_better（将来用）
            if delta > 0:
                improved = True
            elif delta < 0:
                improved = False
            else:
                improved = None

        if improved is True:
            improved_count += 1
        elif improved is False:
            degraded_count += 1
        else:
            unchanged_count += 1

        metrics[key] = {
            "before": bval,
            "after": aval,
            "delta": round(delta, 6),
            "delta_pct": round(delta_pct, 2),
            "improved": improved,
            "unit": unit,
        }

    return {
        "metrics": metrics,
        "summary": {
            "improved_count": improved_count,
            "degraded_count": degraded_count,
            "unchanged_count": unchanged_count,
            "total_metrics": len(metrics),
        },
        "skipped": skipped,
    }


def evaluate_gonogo(
    comparison: dict,
    criteria: dict | None = None,
    latency_result: dict | None = None,
) -> dict:
    """Go/No-Go判定を実行する。

    Args:
        comparison: compare_results()の出力
        criteria: 判定基準（Noneの場合DEFAULT_GONOGO_CRITERIA使用）
        latency_result: レイテンシ測定結果。
            ``{"value": 150.0, "unit": "ms"}`` のような辞書を想定。

    Returns:
        dict with structure::

            {
                "criteria": {
                    "mcd": {
                        "label": "MCD改善",
                        "required": "5%以上改善",
                        "actual": "-2.63%",
                        "status": "FAIL"
                    },
                    ...
                },
                "overall": "FAIL",
                "go_count": 0,
                "total_count": 3,
            }
    """
    if criteria is None:
        criteria = DEFAULT_GONOGO_CRITERIA

    comp_metrics = comparison.get("metrics", {})
    results = {}
    go_count = 0
    total_count = 0

    for crit_key, crit in criteria.items():
        metric_name = crit["metric"]
        direction = crit["direction"]
        label = crit["label"]
        total_count += 1

        # --- レイテンシ（別系統の測定結果） ---
        if direction == "below":
            threshold_abs = crit["threshold_abs"]
            required_str = f"{threshold_abs:.0f}ms以下"

            if latency_result is None:
                results[crit_key] = {
                    "label": label,
                    "required": required_str,
                    "actual": "未測定",
                    "status": "SKIP",
                }
                continue

            lat_value = latency_result.get("value", float("inf"))
            actual_str = f"{lat_value:.1f}ms"
            passed = lat_value <= threshold_abs
            status = "GO" if passed else "FAIL"
            if passed:
                go_count += 1

            results[crit_key] = {
                "label": label,
                "required": required_str,
                "actual": actual_str,
                "status": status,
            }
            continue

        # --- 通常メトリクス (decrease) ---
        threshold_pct = crit["threshold_pct"]
        required_str = f"{threshold_pct:.0f}%以上改善"

        if metric_name not in comp_metrics:
            results[crit_key] = {
                "label": label,
                "required": required_str,
                "actual": "データなし",
                "status": "SKIP",
            }
            continue

        m = comp_metrics[metric_name]
        delta_pct = m["delta_pct"]

        # lower_is_better のメトリクスでは delta_pct が負 = 改善
        # 改善率は絶対値で表現
        if direction == "decrease":
            improvement_pct = -delta_pct  # 減少が正の改善
            actual_str = f"{delta_pct:+.2f}%"
            passed = improvement_pct >= threshold_pct
        else:
            # direction == "increase"（将来用）
            improvement_pct = delta_pct
            actual_str = f"{delta_pct:+.2f}%"
            passed = improvement_pct >= threshold_pct

        status = "GO" if passed else "FAIL"
        if passed:
            go_count += 1

        results[crit_key] = {
            "label": label,
            "required": required_str,
            "actual": actual_str,
            "status": status,
        }

    # SKIPは判定対象から除外して overall を決める
    evaluated = [r for r in results.values() if r["status"] != "SKIP"]
    if not evaluated:
        overall = "SKIP"
    elif all(r["status"] == "GO" for r in evaluated):
        overall = "GO"
    else:
        overall = "FAIL"

    return {
        "criteria": results,
        "overall": overall,
        "go_count": go_count,
        "total_count": total_count,
    }


def format_report(comparison: dict, gonogo: dict) -> str:
    """Markdown形式の比較レポートを生成する。

    Args:
        comparison: compare_results()の出力
        gonogo: evaluate_gonogo()の出力

    Returns:
        Markdown文字列
    """
    lines = []
    lines.append("# RVC 評価ベースライン比較レポート")
    lines.append("")

    # --- メトリクス比較テーブル ---
    lines.append("## メトリクス比較")
    lines.append("")
    lines.append("| メトリクス | Before | After | Delta | 変化率 | 判定 |")
    lines.append("|:-----------|-------:|------:|------:|-------:|:----:|")

    for key, m in comparison["metrics"].items():
        if m["improved"] is True:
            judgment = "改善"
        elif m["improved"] is False:
            judgment = "悪化"
        else:
            judgment = "変化なし"

        unit = m["unit"]
        lines.append(
            f"| {key} | {m['before']:.4f} {unit} "
            f"| {m['after']:.4f} {unit} "
            f"| {m['delta']:+.4f} "
            f"| {m['delta_pct']:+.2f}% "
            f"| {judgment} |"
        )

    lines.append("")

    summary = comparison["summary"]
    lines.append(
        f"改善: {summary['improved_count']} / "
        f"悪化: {summary['degraded_count']} / "
        f"変化なし: {summary['unchanged_count']} / "
        f"合計: {summary['total_metrics']}"
    )

    if comparison.get("skipped"):
        lines.append("")
        lines.append(
            f"スキップされたメトリクス（片方にのみ存在）: "
            f"{', '.join(comparison['skipped'])}"
        )

    lines.append("")

    # --- Go/No-Go 判定テーブル ---
    lines.append("## Go/No-Go 判定")
    lines.append("")
    lines.append("| 基準 | 要件 | 実測 | 結果 |")
    lines.append("|:-----|:-----|:-----|:----:|")

    for _key, c in gonogo["criteria"].items():
        lines.append(
            f"| {c['label']} | {c['required']} | {c['actual']} | {c['status']} |"
        )

    lines.append("")
    lines.append(
        f"**総合判定: {gonogo['overall']}** "
        f"(GO: {gonogo['go_count']}/{gonogo['total_count']})"
    )
    lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Compare two RVC evaluation results and evaluate Go/No-Go criteria.",
    )
    parser.add_argument(
        "--before",
        required=True,
        help="Path to before (baseline) evaluation JSON",
    )
    parser.add_argument(
        "--after",
        required=True,
        help="Path to after (improved) evaluation JSON",
    )
    parser.add_argument(
        "--latency",
        default=None,
        help="Path to latency measurement JSON",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="markdown",
        dest="output_format",
        help="Output format (default: markdown)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # 評価結果の読み込み
    before = load_eval_result(args.before)
    after = load_eval_result(args.after)

    # レイテンシ測定結果（あれば）
    latency_result = None
    if args.latency:
        with open(args.latency, encoding="utf-8") as f:
            latency_result = json.load(f)

    # 比較 & Go/No-Go判定
    comparison = compare_results(before, after)
    gonogo = evaluate_gonogo(comparison, latency_result=latency_result)

    # 出力
    if args.output_format == "json":
        output_data = {
            "comparison": comparison,
            "gonogo": gonogo,
        }
        result_str = json.dumps(output_data, indent=2, ensure_ascii=False)
    else:
        result_str = format_report(comparison, gonogo)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result_str)
        logger.info("結果を %s に出力しました", args.output)
    else:
        print(result_str)


if __name__ == "__main__":
    main()
