import argparse
import os
import sys
from typing import TypedDict, cast

now_dir = os.getcwd()
sys.path.append(now_dir)
from dotenv import load_dotenv
from scipy.io import wavfile

from configs.config import Config
from infer.lib.audio import load_audio
from infer.modules.vc.modules import VC
from lib.types.f0 import PITCH_METHODS, PitchMethod

####
# USAGE
#
# In your Terminal or CMD or whatever


class InferArgs(TypedDict):
    f0up_key: int
    input_path: str
    index_path: str | None
    f0method: PitchMethod
    opt_path: str
    model_name: str
    index_rate: float
    device: str | None
    is_half: bool | None
    filter_radius: int
    resample_sr: int
    rms_mix_rate: float
    protect: float


def arg_parse() -> InferArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--f0up_key", type=int, default=0)
    parser.add_argument("--input_path", type=str, required=True, help="input path")
    parser.add_argument("--index_path", type=str, help="index path")
    parser.add_argument(
        "--f0method",
        type=str,
        choices=PITCH_METHODS,
        default="harvest",
        help="harvest or pm",
    )
    parser.add_argument("--opt_path", type=str, required=True, help="opt path")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="store in assets/weight_root",
    )
    parser.add_argument("--index_rate", type=float, default=0.66, help="index rate")
    parser.add_argument("--device", type=str, help="device")
    parser.add_argument("--is_half", type=bool, help="use half -> True")
    parser.add_argument("--filter_radius", type=int, default=3, help="filter radius")
    parser.add_argument("--resample_sr", type=int, default=0, help="resample sr")
    parser.add_argument("--rms_mix_rate", type=float, default=1, help="rms mix rate")
    parser.add_argument("--protect", type=float, default=0.33, help="protect")

    args = parser.parse_args()
    sys.argv = sys.argv[:1]

    return {
        "f0up_key": args.f0up_key,
        "input_path": args.input_path,
        "index_path": args.index_path,
        "f0method": cast(PitchMethod, args.f0method),
        "opt_path": args.opt_path,
        "model_name": args.model_name,
        "index_rate": args.index_rate,
        "device": args.device,
        "is_half": args.is_half,
        "filter_radius": args.filter_radius,
        "resample_sr": args.resample_sr,
        "rms_mix_rate": args.rms_mix_rate,
        "protect": args.protect,
    }


def main() -> None:
    load_dotenv()
    args = arg_parse()
    config = Config()
    config.device = args["device"] if args["device"] else config.device
    config.is_half = args["is_half"] if args["is_half"] is not None else config.is_half
    vc = VC(config)
    vc.get_vc(args["model_name"])
    audio = load_audio(args["input_path"], 16000)
    message, wav_opt = vc.vc_single(
        (16000, audio),
        args["f0up_key"],
        args["f0method"],
        args["index_path"],
        args["index_rate"],
        args["resample_sr"],
        args["rms_mix_rate"],
        args["protect"],
    )
    if wav_opt is None:
        raise RuntimeError(message)
    wavfile.write(args["opt_path"], wav_opt[0], wav_opt[1])


if __name__ == "__main__":
    main()
