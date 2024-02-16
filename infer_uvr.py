import os
import sys
import argparse
from dotenv import load_dotenv

now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()

from infer.modules.uvr5.modules import uvr


def main(args):

    # model_choose = "HP2_all_vocals"

    # dir_wav_input = ""
    # opt_vocal_root = "opt"

    # wav_inputs = (
    #     "/home/seongwoon/Retrieval-based-Voice-Conversion-WebUI/data/phs/phs_7_04.wav"
    # )

    # opt_ins_root = "opt"
    # format0 = "wav"
    # agg = 10

    output = uvr(
        args.model_choose,
        args.dir_wav_input,
        args.opt_vocal_root,
        args.wav_inputs,
        args.opt_ins_root,
        args.agg,
        args.format0,
    )

    # next(output)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Example argparse script")
    parser.add_argument(
        "--model_choose",
        default="HP2_all_vocals",
        help="uvr5_weights에 위치",
    )
    parser.add_argument(
        "--dir_wav_input",
        default="/home/seongwoon/Retrieval-based-Voice-Conversion-WebUI/opt2",
        help="파일 위치 경로",
    )
    parser.add_argument(
        "--opt_vocal_root",
        default="opt",
        help="vocal 결과 파일 저장 위치",
    )
    parser.add_argument(
        "--wav_inputs",
        default="/home/seongwoon/Retrieval-based-Voice-Conversion-WebUI/data/phs/phs_7_04.wav",
        help="wav 파일 경로",
    )
    parser.add_argument(
        "--opt_ins_root",
        default="opt",
        help="ins 결과 파일 저장 위치",
    )
    parser.add_argument(
        "--agg",
        default=10,
        help="max=20, min=0",
    )
    parser.add_argument(
        "--format0",
        default="wav",
        help="file format",
    )
    args = parser.parse_args()

    main(args)

    # main()
