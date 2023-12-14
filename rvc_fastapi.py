import argparse
import os
import sys

from infer.modules.vc.modules import VC

now_dir = os.getcwd()
sys.path.append(now_dir)
print(now_dir)

from dotenv import load_dotenv
from scipy.io import wavfile

from configs.config import Config


####
# USAGE
#
# In your Terminal or CMD or whatever


def arg_parse() -> tuple:
    parser = argparse.ArgumentParser()
    parser.add_argument("--f0up_key", type=int, default=0)
    parser.add_argument("--input_path", type=str, help="input path")
    parser.add_argument("--index_path", type=str, help="index path")
    parser.add_argument("--f0method", type=str, default="harvest", help="harvest or pm")
    parser.add_argument("--opt_path", type=str, help="opt path")
    parser.add_argument("--model_name", type=str, help="store in assets/weight_root")
    parser.add_argument("--index_rate", type=float, default=0.66, help="index rate")
    parser.add_argument("--device", type=str, help="device")
    parser.add_argument("--is_half", type=bool, help="use half -> True")
    parser.add_argument("--filter_radius", type=int, default=3, help="filter radius")
    parser.add_argument("--resample_sr", type=int, default=0, help="resample sr")
    parser.add_argument("--rms_mix_rate", type=float, default=1, help="rms mix rate")
    parser.add_argument("--protect", type=float, default=0.33, help="protect")

    args = parser.parse_args()
    sys.argv = sys.argv[:1]

    return args




load_dotenv()
args = arg_parse()
#args.input_path = "A:\projects\synth_voice\output\en_speaker_3_simple.wav"
#args.index_path = "A:/projects/Retrieval-based-Voice-Conversion-WebUI/logs/redcliff_clone/added_IVF1254_Flat_nprobe_1_redcliff_clone_v2.index"
#args.model_name = "redcliff_clone.pth"
#args.opt_path = "A:\projects\synth_voice\output\en_speaker_3_simple_out.wav"
config = Config(config_file_folder="A:/projects/Retrieval-based-Voice-Conversion-WebUI/configs/")
config.device = args.device if args.device else config.device
config.is_half = args.is_half if args.is_half else config.is_half
vc = VC(config)
vc.get_vc(args.model_name)
_, wav_opt = vc.vc_single(
    0,
    args.input_path,
    args.f0up_key,
    None,
    args.f0method,
    args.index_path,
    None,
    args.index_rate,
    args.filter_radius,
    args.resample_sr,
    args.rms_mix_rate,
    args.protect,
)
wavfile.write(args.opt_path, wav_opt[0], wav_opt[1])

