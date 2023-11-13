import argparse
import os
import sys
from dotenv import load_dotenv

now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()
from infer.modules.vc.modules import VC

from configs.config import Config
import torch
import warnings
import shutil
import logging

logging.getLogger("numba").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/infer_pack" % (now_dir), ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/uvr5_pack" % (now_dir), ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "assets/weights"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)

config = Config()
vc = VC(config)

def arg_parse() -> tuple:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fO_file", type=int, default=0)
    parser.add_argument("--input_audio_path", type=str, help="input path")
    parser.add_argument("--index_path", type=str, help="index path")
    parser.add_argument("--f0_method", type=str, default="harvest", help="harvest or pm")
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

opt_path = "test_v2.wav"

fO_file = None
f0_method = 'rmvpe'
fO_up_key = 0.0
file_index = ''
file_index2 = 'logs/barv/added_IVF3462_Flat_nprobe_1_barv_v2.index'
filter_radius = 3
index_rate = 0.75
input_audio_path = '/Users/roman/Downloads/rvc_audio_test/K_7.aif'
protect = 0.33
resample_sr = 0
rms_mix_rate = 0.25
sid = 0

from scipy.io import wavfile

vc.get_vc("barv.pth")
_, wav_opt = vc.vc_single(
    sid,
    input_audio_path,
    fO_up_key,
    fO_file,
    f0_method,
    file_index,
    file_index2,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect)

wavfile.write(opt_path, wav_opt[0], wav_opt[1])

if os.path.isfile(opt_path):
    file_stats = os.stat(opt_path)
    if file_stats.st_size > 0:
        print("SUCCESS")
    else:
        print("FAILURE")
else:
    print("FAILURE")
