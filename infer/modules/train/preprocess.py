import multiprocessing
import os
import sys
from pathlib import Path
from typing import List, Tuple, cast

from scipy import signal

now_dir = os.getcwd()
sys.path.append(now_dir)
print(*sys.argv[1:])
inp_root = sys.argv[1]
sr = int(sys.argv[2])
n_p = int(sys.argv[3])
exp_dir = Path(sys.argv[4])
noparallel = sys.argv[5] == "True"
per = float(sys.argv[6])
import os
import traceback

import librosa
import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile

from infer.lib.audio import load_audio
from infer.lib.slicer2 import Slicer

f = open(exp_dir / "preprocess.log", "a+")


def println(strr):
    print(strr)
    f.write("%s\n" % strr)
    f.flush()


class PreProcess:
    slicer: Slicer
    sr: int
    bh: NDArray[np.floating]
    ah: NDArray[np.floating]
    per: float
    overlap: float
    tail: float
    max: float
    alpha: float
    exp_dir: Path
    gt_wavs_dir: Path
    wavs16k_dir: Path

    def __init__(self: "PreProcess", sr: int, exp_dir: Path, per=3.7):
        self.slicer = Slicer(
            sr=sr,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.sr = sr
        bh, ah = cast(
            tuple[NDArray[np.floating], NDArray[np.floating]],
            signal.butter(N=5, Wn=48, btype="highpass", fs=self.sr, output="ba"),
        )
        self.bh = np.asarray(bh, dtype=np.float64)
        self.ah = np.asarray(ah, dtype=np.float64)
        self.per = per
        self.overlap = 0.3
        self.tail = self.per + self.overlap
        self.max = 0.9
        self.alpha = 0.75
        self.exp_dir: Path = exp_dir
        self.gt_wavs_dir: Path = exp_dir / "0_gt_wavs"
        self.wavs16k_dir: Path = exp_dir / "1_16k_wavs"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.gt_wavs_dir.mkdir(parents=True, exist_ok=True)
        self.wavs16k_dir.mkdir(parents=True, exist_ok=True)

    def norm_write(
        self: "PreProcess", tmp_audio: np.ndarray, idx0: int, idx1: int
    ) -> None:
        tmp_max = np.abs(tmp_audio).max()
        if tmp_max > 2.5:
            print("%s-%s-%s-filtered" % (idx0, idx1, tmp_max))
            return
        tmp_audio = (tmp_audio / tmp_max * (self.max * self.alpha)) + (
            1 - self.alpha
        ) * tmp_audio
        wavfile.write(
            str(self.gt_wavs_dir / f"{idx0}_{idx1}.wav"),
            self.sr,
            tmp_audio.astype(np.float32),
        )
        tmp_audio = librosa.resample(
            tmp_audio, orig_sr=self.sr, target_sr=16000
        )  # , res_type="soxr_vhq"
        wavfile.write(
            str(self.wavs16k_dir / f"{idx0}_{idx1}.wav"),
            16000,
            tmp_audio.astype(np.float32),
        )

    def pipeline(self: "PreProcess", path: str, idx0: int):
        try:
            audio = load_audio(path, self.sr)
            # zero phased digital filter cause pre-ringing noise...
            # audio = signal.filtfilt(self.bh, self.ah, audio)
            audio = signal.lfilter(self.bh, self.ah, audio)

            idx1 = 0
            for audio in self.slicer.slice(audio):
                i = 0
                while 1:
                    start = int(self.sr * (self.per - self.overlap) * i)
                    i += 1
                    if len(audio[start:]) > self.tail * self.sr:
                        tmp_audio = audio[start : start + int(self.per * self.sr)]
                        self.norm_write(tmp_audio, idx0, idx1)
                        idx1 += 1
                    else:
                        tmp_audio = audio[start:]
                        idx1 += 1
                        break
                self.norm_write(tmp_audio, idx0, idx1)
            println("%s\t-> Success" % path)
        except:
            println("%s\t-> %s" % (path, traceback.format_exc()))

    def pipeline_mp(self: "PreProcess", infos: List[Tuple[str, int]]) -> None:
        for path, idx0 in infos:
            self.pipeline(path, idx0)

    def pipeline_mp_inp_dir(self: "PreProcess", inp_root: str, n_p: int) -> None:
        try:
            inp_root_path = Path(inp_root)
            infos = [
                (str(inp_root_path / name), idx)
                for idx, name in enumerate(sorted(inp_root_path.iterdir(), key=lambda p: p.name))
            ]
            if noparallel:
                for i in range(n_p):
                    self.pipeline_mp(infos[i::n_p])
            else:
                ps = []
                for i in range(n_p):
                    p = multiprocessing.Process(
                        target=self.pipeline_mp, args=(infos[i::n_p],)
                    )
                    ps.append(p)
                    p.start()
                for i in range(n_p):
                    ps[i].join()
        except:
            println("Fail. %s" % traceback.format_exc())


def preprocess_trainset(inp_root: str, sr: int, n_p: int, exp_dir: Path, per: float):
    pp = PreProcess(sr, exp_dir, per)
    println("start preprocess")
    pp.pipeline_mp_inp_dir(inp_root, n_p)
    println("end preprocess")


if __name__ == "__main__":
    preprocess_trainset(inp_root, sr, n_p, exp_dir, per)
