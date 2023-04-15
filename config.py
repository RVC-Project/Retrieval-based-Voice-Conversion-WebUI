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
listen_port = cmd_opts.port
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

if device not in ["cpu", "mps"]:
    gpu_name = torch.cuda.get_device_name(int(device.split(":")[-1]))
    if "16" in gpu_name or "MX" in gpu_name:
        print("16系显卡/MX系显卡强制单精度")
        is_half = False

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
