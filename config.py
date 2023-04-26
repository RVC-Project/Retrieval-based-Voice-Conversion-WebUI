########################硬件参数########################

# 填写cuda:x, cpu 或 mps, x指代第几张卡，只支持 N卡 / Apple Silicon 加速
device = "cuda:0"

# 9-10-20-30-40系显卡无脑True，不影响质量，>=20显卡开启有加速
is_half = True

# 默认0用上所有线程，写数字限制CPU资源使用
n_cpu = 0

########################硬件参数########################


##################下为参数处理逻辑，勿动##################

########################命令行参数########################
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=7865, help="Listen port")
parser.add_argument("--pycmd", type=str, default="python", help="Python command")
parser.add_argument("--colab", action="store_true", help="Launch in colab")
parser.add_argument(
    "--noparallel", action="store_true", help="Disable parallel processing"
)
parser.add_argument(
    "--noautoopen", action="store_true", help="Do not open in browser automatically"
)
cmd_opts = parser.parse_args()

python_cmd = cmd_opts.pycmd
listen_port = cmd_opts.port if 0 <= cmd_opts.port <= 65535 else 7865
iscolab = cmd_opts.colab
noparallel = cmd_opts.noparallel
noautoopen = cmd_opts.noautoopen
########################命令行参数########################

import sys
import torch


# has_mps is only available in nightly pytorch (for now) and MasOS 12.3+.
# check `getattr` and try it for compatibility
def has_mps() -> bool:
    if sys.platform != "darwin":
        return False
    else:
        if not getattr(torch, "has_mps", False):
            return False
        try:
            torch.zeros(1).to(torch.device("mps"))
            return True
        except Exception:
            return False


if not torch.cuda.is_available():
    if has_mps():
        print("没有发现支持的N卡, 使用MPS进行推理")
        device = "mps"
    else:
        print("没有发现支持的N卡, 使用CPU进行推理")
        device = "cpu"
        is_half = False

gpu_mem=None
if device not in ["cpu", "mps"]:
    i_device=int(device.split(":")[-1])
    gpu_name = torch.cuda.get_device_name(i_device)
    if "16" in gpu_name or "P40"in gpu_name.upper() or "1070"in gpu_name or "1080"in gpu_name:
        print("16系显卡强制单精度")
        is_half = False
        with open("configs/32k.json","r")as f:strr=f.read().replace("true","false")
        with open("configs/32k.json","w")as f:f.write(strr)
        with open("configs/40k.json","r")as f:strr=f.read().replace("true","false")
        with open("configs/40k.json","w")as f:f.write(strr)
        with open("configs/48k.json","r")as f:strr=f.read().replace("true","false")
        with open("configs/48k.json","w")as f:f.write(strr)
        with open("trainset_preprocess_pipeline_print.py","r")as f:strr=f.read().replace("3.7","3.0")
        with open("trainset_preprocess_pipeline_print.py","w")as f:f.write(strr)
    gpu_mem=int(torch.cuda.get_device_properties(i_device).total_memory/1024/1024/1024+0.4)
    if(gpu_mem<=4):
        with open("trainset_preprocess_pipeline_print.py","r")as f:strr=f.read().replace("3.7","3.0")
        with open("trainset_preprocess_pipeline_print.py","w")as f:f.write(strr)
from multiprocessing import cpu_count

if n_cpu == 0:
    n_cpu = cpu_count()
if is_half:
    # 6G显存配置
    x_pad = 3
    x_query = 10
    x_center = 60
    x_max = 65
else:
    # 5G显存配置
    x_pad = 1
    x_query = 6
    x_center = 38
    x_max = 41
if(gpu_mem!=None and gpu_mem<=4):
    x_pad = 1
    x_query = 5
    x_center = 30
    x_max = 32