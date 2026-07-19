import multiprocessing
import os
import sys

from scipy import signal

inp_root = sys.argv[1]
sr = int(sys.argv[2])
n_p = int(sys.argv[3])
exp_dir = sys.argv[4]
noparallel = sys.argv[5] == "True"
per = float(sys.argv[6])
import os
import traceback

import librosa
import numpy as np
from scipy.io import wavfile

from infer.audio import load_audio
from train.dataset.slicer2 import Slicer
from i18n.i18n import I18nAuto
from tools.progress import should_report

i18n = I18nAuto()

f = open("%s/preprocess.log" % exp_dir, "a", encoding="utf8")


def println(strr):
    print(strr)
    f.write("%s\n" % strr)
    f.flush()


class PreProcess:
    def __init__(self, sr, exp_dir, per=3.7):
        self.slicer = Slicer(
            sr=sr,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.sr = sr
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr)
        self.per = per
        self.overlap = 0.3
        self.tail = self.per + self.overlap
        self.max = 0.9
        self.alpha = 0.75
        self.exp_dir = exp_dir
        self.gt_wavs_dir = "%s/0_gt_wavs" % exp_dir
        self.wavs16k_dir = "%s/1_16k_wavs" % exp_dir
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

    def norm_write(self, tmp_audio, idx0, idx1):
        tmp_max = np.abs(tmp_audio).max()
        if not np.isfinite(tmp_max) or tmp_max <= 0 or tmp_max > 2.5:
            println(
                i18n("[数据切分][跳过] 无效或异常音频片段：%s_%s | 峰值：%s")
                % (idx0, idx1, tmp_max)
            )
            return False
        tmp_audio = (tmp_audio / tmp_max * (self.max * self.alpha)) + (
            1 - self.alpha
        ) * tmp_audio
        wavfile.write(
            "%s/%s_%s.wav" % (self.gt_wavs_dir, idx0, idx1),
            self.sr,
            tmp_audio.astype(np.float32),
        )
        tmp_audio = librosa.resample(
            tmp_audio, orig_sr=self.sr, target_sr=16000
        )  # , res_type="soxr_vhq"
        wavfile.write(
            "%s/%s_%s.wav" % (self.wavs16k_dir, idx0, idx1),
            16000,
            tmp_audio.astype(np.float32),
        )
        return True

    def pipeline(self, path, idx0, total):
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
            if should_report(idx0, total):
                println(
                    i18n("[数据切分] 进度：%s/%s | %s")
                    % (idx0 + 1, total, os.path.basename(path))
                )
            return True
        except Exception:
            println(
                i18n("[数据切分][失败] %s\n%s")
                % (path, traceback.format_exc())
            )
            return False

    def pipeline_mp(self, infos):
        success = 0
        failed = 0
        for path, idx0, total in infos:
            if self.pipeline(path, idx0, total):
                success += 1
            else:
                failed += 1
        if infos:
            println(
                i18n("[数据切分] 子任务完成 | 成功：%s | 失败：%s")
                % (success, failed)
            )

    def pipeline_mp_inp_dir(self, inp_root, n_p):
        try:
            names = sorted(os.listdir(inp_root))
            total = len(names)
            infos = [
                ("%s/%s" % (inp_root, name), idx, total)
                for idx, name in enumerate(names)
            ]
            println(i18n("[数据切分] 待处理：%s | 进程数：%s") % (total, n_p))
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
        except Exception:
            println(i18n("[数据切分][失败] %s") % traceback.format_exc())


def preprocess_trainset(inp_root, sr, n_p, exp_dir, per):
    pp = PreProcess(sr, exp_dir, per)
    println(i18n("[数据切分] 开始"))
    pp.pipeline_mp_inp_dir(inp_root, n_p)
    println(i18n("[数据切分] 完成"))


if __name__ == "__main__":
    preprocess_trainset(inp_root, sr, n_p, exp_dir, per)
