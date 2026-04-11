"""M1改善前ベースライン構成の復元と比較ツール。

Usage:
    # Step 1: Pre-M1設定ファイルを生成
    uv run python tools/eval/run_m1_baseline.py generate-config

    # Step 2: Pre-M1設定で学習（手動実行が必要）
    uv run python tools/eval/run_m1_baseline.py show-commands --exp-name haruna_pre_m1

    # Step 3: 結果の比較
    uv run python tools/eval/run_m1_baseline.py compare \
        --before eval_output/baseline_pre_m1.json \
        --after eval_output/baseline_e200.json
"""

import argparse
import copy
import json
import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# M1で変更されたパラメータと元の値
M1_CHANGES = {
    "train": {
        "segment_size": {"pre_m1": 17280, "m1": 34560},
        "weight_decay": {"pre_m1": None, "m1": 0.01},  # None = キーなし
        "c_mrstft": {"pre_m1": None, "m1": 5.0},
        "bf16_run": {"pre_m1": None, "m1": True},
    },
    "model": {
        "p_dropout": {"pre_m1": 0, "m1": 0.1},
    },
}


def generate_pre_m1_config(
    source_config: str = "configs/v2/48k.json",
    output_config: str = "configs/v2/48k_pre_m1.json",
) -> str:
    """現在の48k.jsonからM1変更を除去したpre-M1設定を生成する。

    Args:
        source_config: ソース設定ファイルのパス（プロジェクトルートからの相対パスまたは絶対パス）
        output_config: 出力設定ファイルのパス（プロジェクトルートからの相対パスまたは絶対パス）

    Returns:
        出力ファイルの絶対パス
    """
    # 絶対パスに変換
    if not os.path.isabs(source_config):
        source_config = os.path.join(_PROJECT_ROOT, source_config)
    if not os.path.isabs(output_config):
        output_config = os.path.join(_PROJECT_ROOT, output_config)

    with open(source_config, encoding="utf-8") as f:
        config = json.load(f)

    pre_m1 = copy.deepcopy(config)

    # M1_CHANGESの内容を元の値に戻す
    for section, params in M1_CHANGES.items():
        if section not in pre_m1:
            continue
        for key, values in params.items():
            pre_m1_value = values["pre_m1"]
            if pre_m1_value is None:
                # M1で追加されたキーを削除
                pre_m1[section].pop(key, None)
            else:
                # M1で変更された値を元に戻す
                pre_m1[section][key] = pre_m1_value

    os.makedirs(os.path.dirname(output_config), exist_ok=True)
    with open(output_config, "w", encoding="utf-8") as f:
        json.dump(pre_m1, f, indent=2, ensure_ascii=False)
        f.write("\n")

    logger.info("Pre-M1設定を生成しました: %s", output_config)

    # 変更内容のサマリーを表示
    print(f"生成完了: {output_config}")
    print()
    print("M1からの復元内容:")
    for section, params in M1_CHANGES.items():
        for key, values in params.items():
            m1_val = values["m1"]
            pre_val = values["pre_m1"]
            if pre_val is None:
                print(f"  [{section}] {key}: {m1_val} -> (削除)")
            else:
                print(f"  [{section}] {key}: {m1_val} -> {pre_val}")

    return output_config


def show_training_commands(exp_name: str = "baseline_pre_m1") -> str:
    """Pre-M1設定で学習するためのコマンドを表示する。

    Args:
        exp_name: 実験名

    Returns:
        コマンド文字列
    """
    commands = f"""\
# === M1改善前ベースライン学習手順 ===
#
# 1. Pre-M1設定で前処理・学習を実行:
#    前処理は現在のコードで問題ありません（M1は学習設定のみの変更）
#
#    学習コマンド（configs/v2/48k_pre_m1.jsonを使用）:

# Step 1: 設定ファイルをコピー
cp configs/v2/48k_pre_m1.json logs/{exp_name}/config.json

# Step 2: ファイルリストを作成（既存のログディレクトリから流用可能）
cp logs/haruna_singing/filelist.txt logs/{exp_name}/filelist.txt

# Step 3: 前処理済みデータをシンボリックリンク（再処理不要）
# 前処理はM1の影響を受けないため、既存データを流用
ln -s $(pwd)/logs/haruna_singing/0_gt_wavs logs/{exp_name}/0_gt_wavs
ln -s $(pwd)/logs/haruna_singing/1_16k_wavs logs/{exp_name}/1_16k_wavs
ln -s $(pwd)/logs/haruna_singing/2a_f0 logs/{exp_name}/2a_f0
ln -s $(pwd)/logs/haruna_singing/2b-f0nsf logs/{exp_name}/2b-f0nsf
ln -s $(pwd)/logs/haruna_singing/3_feature768 logs/{exp_name}/3_feature768

# Step 4: 学習実行
PYTHONUTF8=1 uv run python _run_train.py \\
    -e {exp_name} \\
    -sr 48k \\
    -f0 1 \\
    -bs 8 \\
    -te 200 \\
    -se 50 \\
    -pg assets/pretrained_v2/f0G48k.pth \\
    -pd assets/pretrained_v2/f0D48k.pth \\
    -l 0 -c 0 -sw 0 -v v2

# Step 5: 推論実行
uv run python tools/infer_cli.py \\
    --model_name {exp_name}.pth \\
    --input_path "C:\\Users\\yuta\\Desktop\\AIHUB\\はるなさん歌声\\CatLoving_Vox.wav" \\
    --opt_path eval_output/CatLoving_conv_pre_m1.wav \\
    --index_path logs/{exp_name}/added_*.index \\
    --f0method rmvpe

# Step 6: 評価実行
uv run python tools/eval/run_eval.py \\
    --ref "C:\\Users\\yuta\\Desktop\\AIHUB\\はるなさん歌声\\CatLoving_Vox.wav" \\
    --conv eval_output/CatLoving_conv_pre_m1.wav \\
    --metrics all \\
    --device cuda \\
    --output eval_output/baseline_pre_m1.json

# Step 7: 比較
uv run python tools/eval/baseline_compare.py \\
    --before eval_output/baseline_pre_m1.json \\
    --after eval_output/baseline_e200.json"""
    return commands


def run_compare(before_path: str, after_path: str) -> None:
    """baseline_compare.pyを呼び出して2つの評価結果を比較する。

    Args:
        before_path: 改善前の評価結果JSONパス
        after_path: 改善後の評価結果JSONパス
    """
    compare_script = os.path.join(_PROJECT_ROOT, "tools", "eval", "baseline_compare.py")
    cmd = [
        sys.executable,
        compare_script,
        "--before", before_path,
        "--after", after_path,
    ]
    logger.info("実行: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="M1改善前ベースライン復元・比較ツール",
    )
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("generate-config", help="Pre-M1設定ファイルを生成")
    gen.add_argument(
        "--source", default="configs/v2/48k.json",
        help="ソース設定ファイルパス (default: configs/v2/48k.json)",
    )
    gen.add_argument(
        "--output", default="configs/v2/48k_pre_m1.json",
        help="出力設定ファイルパス (default: configs/v2/48k_pre_m1.json)",
    )

    show = sub.add_parser("show-commands", help="学習コマンドを表示")
    show.add_argument(
        "--exp-name", default="baseline_pre_m1",
        help="実験名 (default: baseline_pre_m1)",
    )

    cmp = sub.add_parser("compare", help="結果を比較")
    cmp.add_argument("--before", required=True, help="改善前の評価結果JSON")
    cmp.add_argument("--after", required=True, help="改善後の評価結果JSON")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command == "generate-config":
        generate_pre_m1_config(
            source_config=args.source,
            output_config=args.output,
        )
    elif args.command == "show-commands":
        print(show_training_commands(exp_name=args.exp_name))
    elif args.command == "compare":
        run_compare(before_path=args.before, after_path=args.after)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
