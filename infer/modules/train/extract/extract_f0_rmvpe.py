import os
import sys
import traceback

import parselmouth
from tap import Tap

now_dir = os.getcwd()
sys.path.append(now_dir)
import logging

import numpy as np
import pyworld

from infer.lib.audio import load_audio


class ExtractF0RmvpeArgs(Tap):
    n_part: int
    i_part: int
    i_gpu: str
    exp_dir: str
    is_half: bool

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
is_half = args.is_half
f = open(f"{exp_dir}/extract_f0_feature.log", "a+")


def printt(strr):
    print(strr)
    f.write(f"{strr}\n")
    f.flush()


class FeatureInput:
    def __init__(self, samplerate=16000, hop_size=160):
        self.fs = samplerate
        self.hop = hop_size

        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    def compute_f0(self, path, f0_method):
        x = load_audio(path, self.fs)
        # p_len = x.shape[0] // self.hop
        if f0_method == "rmvpe":
            if hasattr(self, "model_rmvpe") == False:
                from infer.lib.rmvpe import RMVPE

                print("Loading rmvpe model")
                self.model_rmvpe = RMVPE(
                    "assets/rmvpe/rmvpe.pt", is_half=is_half, device="cuda"
                )
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        else:
            raise ValueError(f"Unsupported f0 method: {f0_method}")
        return f0

    def coarse_f0(self, f0):
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
            self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1

        # use 0 or 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(int)
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
            f0_coarse.max(),
            f0_coarse.min(),
        )
        return f0_coarse

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
                    featur_pit = self.compute_f0(inp_path, f0_method)
                    np.save(
                        opt_path2,
                        featur_pit,
                        allow_pickle=False,
                    )  # nsf
                    coarse_pit = self.coarse_f0(featur_pit)
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
