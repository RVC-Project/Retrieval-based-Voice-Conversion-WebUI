import os
import sys
import traceback
from pathlib import Path
from typing import Protocol, cast

import parselmouth
from tap import Tap

now_dir = os.getcwd()
sys.path.append(now_dir)
import logging

import numpy as np
import pyworld

from infer.lib.audio import load_audio
from lib.types.f0 import PitchMethod

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


class PyWorldModule(Protocol):
    def harvest(
        self,
        x: np.ndarray,
        fs: int,
        f0_ceil: float,
        f0_floor: float,
        frame_period: float,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def dio(
        self,
        x: np.ndarray,
        fs: int,
        f0_ceil: float,
        f0_floor: float,
        frame_period: float,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def stonemask(
        self, x: np.ndarray, f0: np.ndarray, t: np.ndarray, fs: int
    ) -> np.ndarray: ...


pyworld_api = cast(PyWorldModule, pyworld)


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

        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    def compute_f0(self, path, f0_method):
        x = load_audio(path, self.fs)
        p_len = x.shape[0] // self.hop
        if f0_method == "pm":
            time_step = 160 / 16000 * 1000
            f0_min = 50
            f0_max = 1100
            f0 = (
                parselmouth.Sound(x, self.fs)
                .to_pitch_ac(
                    time_step=time_step / 1000,
                    voicing_threshold=0.6,
                    pitch_floor=f0_min,
                    pitch_ceiling=f0_max,
                )
                .selected_array["frequency"]
            )
            pad_size = (p_len - len(f0) + 1) // 2
            if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                f0 = np.pad(
                    f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
                )
        elif f0_method == "harvest":
            f0, t = pyworld_api.harvest(
                x.astype(np.double),
                fs=self.fs,
                f0_ceil=self.f0_max,
                f0_floor=self.f0_min,
                frame_period=1000 * self.hop / self.fs,
            )
            f0 = pyworld_api.stonemask(x.astype(np.double), f0, t, self.fs)
        elif f0_method == "dio":
            f0, t = pyworld_api.dio(
                x.astype(np.double),
                fs=self.fs,
                f0_ceil=self.f0_max,
                f0_floor=self.f0_min,
                frame_period=1000 * self.hop / self.fs,
            )
            f0 = pyworld_api.stonemask(x.astype(np.double), f0, t, self.fs)
        elif f0_method == "rmvpe":
            if hasattr(self, "model_rmvpe") == False:
                from infer.lib.rmvpe import RMVPE

                print("Loading rmvpe model")
                self.model_rmvpe = RMVPE(
                    "assets/rmvpe/rmvpe.pt", is_half=False, device="cpu"
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
