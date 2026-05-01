import os
import sys
import traceback
from pathlib import Path
from typing import Literal

from tap import Tap

now_dir = os.getcwd()
sys.path.append(now_dir)
import logging

import numpy as np

from infer.lib.audio import load_audio
from lib.f0 import Generator

BoolString = Literal["True", "False", "true", "false", "1", "0"]


def parse_bool(value: BoolString) -> bool:
    return value.lower() in {"true", "1"}


class ExtractF0RmvpeArgs(Tap):
    # Total number of extraction partitions.
    n_part: int
    # Partition index handled by this worker.
    i_part: int
    # GPU ID assigned to this worker.
    i_gpu: str
    # Experiment directory.
    exp_dir: str
    # Whether to use half precision.
    is_half: BoolString

    def configure(self) -> None:
        self.add_argument("n_part")
        self.add_argument("i_part")
        self.add_argument("i_gpu")
        self.add_argument("exp_dir")
        self.add_argument("is_half")


args = ExtractF0RmvpeArgs().parse_args()
n_part = args.n_part
i_part = args.i_part
i_gpu = args.i_gpu
os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)
exp_dir = args.exp_dir
is_half = parse_bool(args.is_half)
f = open(f"{exp_dir}/extract_f0_feature.log", "a+")


def printt(strr):
    print(strr)
    f.write(f"{strr}\n")
    f.flush()


class FeatureInput:
    def __init__(self, samplerate=16000, hop_size=160):
        self.fs = samplerate
        self.hop = hop_size
        self.f0_gen = Generator(
            Path("assets/rmvpe"),
            is_half,
            0,
            device="cuda",
            window=self.hop,
            sr=self.fs,
        )

    def go(self, paths, f0_method):
        if len(paths) == 0:
            printt("no-f0-todo")
        else:
            printt(f"todo-f0-{len(paths)}")
            n = max(len(paths) // 5, 1)  # print at most 5 lines per process
            for idx, (inp_path, opt_path1, opt_path2) in enumerate(paths):
                try:
                    if idx % n == 0:
                        printt(f"f0ing,now-{idx},all-{len(paths)},-{inp_path}")
                    if (
                        os.path.exists(opt_path1 + ".npy") == True
                        and os.path.exists(opt_path2 + ".npy") == True
                    ):
                        continue
                    audio = load_audio(inp_path, self.fs)
                    p_len = audio.shape[0] // self.hop
                    coarse_pit, featur_pit = self.f0_gen.calculate(
                        audio,
                        p_len,
                        0,
                        "rmvpe",
                        3,
                    )
                    np.save(
                        opt_path2,
                        featur_pit,
                        allow_pickle=False,
                    )  # nsf
                    np.save(
                        opt_path1,
                        coarse_pit,
                        allow_pickle=False,
                    )  # ori
                except:
                    printt("f0fail-%s-%s-%s" % (idx, inp_path, traceback.format_exc()))


if __name__ == "__main__":
    # exp_dir=r"E:\codes\py39\dataset\mi-test"
    # n_p=16
    # f = open("%s/log_extract_f0.log"%exp_dir, "w")
    printt(" ".join(sys.argv))
    featureInput = FeatureInput()
    paths = []
    inp_root = "%s/1_16k_wavs" % (exp_dir)
    opt_root1 = "%s/2a_f0" % (exp_dir)
    opt_root2 = "%s/2b-f0nsf" % (exp_dir)

    os.makedirs(opt_root1, exist_ok=True)
    os.makedirs(opt_root2, exist_ok=True)
    for name in sorted(list(os.listdir(inp_root))):
        inp_path = "%s/%s" % (inp_root, name)
        if "spec" in inp_path:
            continue
        opt_path1 = "%s/%s" % (opt_root1, name)
        opt_path2 = "%s/%s" % (opt_root2, name)
        paths.append([inp_path, opt_path1, opt_path2])
    try:
        featureInput.go(paths[i_part::n_part], "rmvpe")
    except:
        printt("f0_all_fail-%s" % (traceback.format_exc()))
    # ps = []
    # for i in range(n_p):
    #     p = Process(
    #         target=featureInput.go,
    #         args=(
    #             paths[i::n_p],
    #             f0method,
    #         ),
    #     )
    #     ps.append(p)
    #     p.start()
    # for i in range(n_p):
    #     ps[i].join()
