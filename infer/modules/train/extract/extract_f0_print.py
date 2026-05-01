import os
import sys
import traceback
from pathlib import Path

from tap import Tap

now_dir = os.getcwd()
sys.path.append(now_dir)
import logging

import numpy as np

from infer.lib.audio import load_audio
from lib.f0 import Generator, PitchMethod

from multiprocessing import Process


class ExtractF0Args(Tap):
    # Experiment directory.
    exp_dir: Path
    # Number of CPU extraction workers.
    n_p: int
    # F0 extraction method.
    f0method: PitchMethod

    def configure(self) -> None:
        self.add_argument("exp_dir")
        self.add_argument("n_p")
        self.add_argument("f0method")


args = ExtractF0Args().parse_args()
exp_dir = args.exp_dir
f = open(exp_dir / "extract_f0_feature.log", "a+")


def printt(strr):
    print(strr)
    f.write(f"{strr}\n")
    f.flush()


n_p = args.n_p
f0method = args.f0method


class FeatureInput:
    def __init__(self, samplerate=16000, hop_size=160):
        self.fs = samplerate
        self.hop = hop_size
        self.f0_gen = Generator(
            Path("assets/rmvpe"),
            False,
            0,
            device="cpu",
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
                        f0_method,
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
    inp_root = exp_dir / "1_16k_wavs"
    opt_root1 = exp_dir / "2a_f0"
    opt_root2 = exp_dir / "2b-f0nsf"

    opt_root1.mkdir(parents=True, exist_ok=True)
    opt_root2.mkdir(parents=True, exist_ok=True)
    for wav_file in sorted(inp_root.iterdir(), key=lambda p: p.name):
        inp_path = inp_root / wav_file.name
        if "spec" in inp_path.name:
            continue
        opt_path1 = opt_root1 / wav_file.name
        opt_path2 = opt_root2 / wav_file.name
        paths.append([str(inp_path), str(opt_path1), str(opt_path2)])

    ps = []
    for i in range(n_p):
        p = Process(
            target=featureInput.go,
            args=(
                paths[i::n_p],
                f0method,
            ),
        )
        ps.append(p)
        p.start()
    for i in range(n_p):
        ps[i].join()
