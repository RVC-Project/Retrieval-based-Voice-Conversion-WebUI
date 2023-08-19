import os

import torch

from config import Config
from i18n import I18nAuto

# Load language
i18n = I18nAuto()
i18n.print()

# Initialize vars
now_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tmp = os.path.join(now_dir, "TEMP")
config = Config()


# Check GPU
def check_gpu(i18n):
    n_gpu = torch.cuda.device_count()
    gpu_infos = []
    mem = []
    if_gpu_ok = False

    if torch.cuda.is_available() or n_gpu != 0:
        for i in range(n_gpu):
            gpu_name = torch.cuda.get_device_name(i)
            if any(
                    value in gpu_name.upper()
                    for value in [
                        "10",
                        "16",
                        "20",
                        "30",
                        "40",
                        "A2",
                        "A3",
                        "A4",
                        "P4",
                        "A50",
                        "500",
                        "A60",
                        "70",
                        "80",
                        "90",
                        "M4",
                        "T4",
                        "TITAN",
                    ]
            ):
                # A10#A100#V100#A40#P40#M40#K80#A4500
                if_gpu_ok = True  # 至少有一张能用的N卡
                gpu_infos.append("%s\t%s" % (i, gpu_name))
                mem.append(
                    int(
                        torch.cuda.get_device_properties(i).total_memory
                        / 1024
                        / 1024
                        / 1024
                        + 0.4
                    )
                )
    if if_gpu_ok and len(gpu_infos) > 0:
        gpu_info = "\n".join(gpu_infos)
        default_batch_size = min(mem) // 2
    else:
        gpu_info = i18n("很遗憾您这没有能用的显卡来支持您训练")
        default_batch_size = 1
    gpus = "-".join([i[0] for i in gpu_infos])

    return gpu_info, default_batch_size, gpus


# Check paths
weight_root = os.path.join(now_dir, "weights")
weight_uvr5_root = os.path.join(now_dir, "uvr5_weights")
index_root = os.path.join(now_dir, "logs")
names = []
index_paths = []
uvr5_names = []
for name in os.listdir(weight_root):
    if name.endswith(".pth"):
        names.append(name)
for root, dirs, files in os.walk(index_root, topdown=False):
    for name in files:
        if name.endswith(".index") and "trained" not in name:
            index_paths.append("%s/%s" % (root, name))
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth") or "onnx" in name:
        uvr5_names.append(name.replace(".pth", ""))


def get_index_path_from_model(sid, index_paths):
    sel_index_path = ""
    name = os.path.join("logs", sid.split(".")[0], "")
    for f in index_paths:
        if name in f:
            sel_index_path = f
            break
    return sel_index_path
