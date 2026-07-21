import os
import sys
import traceback

import parselmouth

import logging

import numpy as np

from i18n.i18n import I18nAuto
from tools.progress import should_report


i18n = I18nAuto()

logging.getLogger("numba").setLevel(logging.WARNING)
from multiprocessing import Process

mode = sys.argv[1].lower()
if mode == "cpu":
    exp_dir = sys.argv[2]
    n_p = int(sys.argv[3])
    f0method = sys.argv[4]
    device = "cpu"
    is_half = False
elif mode == "cuda":
    n_part = int(sys.argv[2])
    i_part = int(sys.argv[3])
    i_gpu = sys.argv[4]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)
    exp_dir = sys.argv[5]
    is_half = sys.argv[6].lower() == "true"
    f0method = "rmvpe"
    device = "cuda"
elif mode in ("dml", "directml"):
    exp_dir = sys.argv[2]
    f0method = "rmvpe"
    is_half = False
    import torch_directml

    device = torch_directml.device(torch_directml.default_device())
else:
    raise ValueError("Unsupported F0 extraction mode: %s" % mode)

# CUDA_VISIBLE_DEVICES must be set before infer.audio imports torch/configs.
from infer.audio import load_audio

f = open("%s/extract_f0_feature.log" % exp_dir, "a", encoding="utf8")


def printt(strr):
    print(strr)
    f.write("%s\n" % strr)
    f.flush()
class FeatureInput(object):
    def __init__(self, samplerate=16000, hop_size=160):
        self.fs = samplerate
        self.hop = hop_size

        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    def compute_f0(self, path, f0_method):
        if f0_method not in ("pm", "rmvpe"):
            raise ValueError(i18n("仅支持pm和rmvpe音高提取算法"))
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
        elif f0_method == "rmvpe":
            if hasattr(self, "model_rmvpe") == False:
                from infer.rmvpe import RMVPE

                printt(i18n("正在加载RMVPE模型"))
                self.model_rmvpe = RMVPE(
                    "assets/rmvpe/rmvpe.pt", is_half=is_half, device=device
                )
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        f0 = np.asarray(f0)
        try:
            uv = f0 == 0
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
        except Exception:
            traceback.print_exc()
            return None
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

    def go(self, paths, f0_method, max_updates=5):
        success = 0
        skipped = 0
        failed = 0
        if len(paths) == 0:
            printt(i18n("[F0提取] 无待处理音频，已全部跳过"))
        else:
            printt(i18n("[F0提取] 待处理：%s") % len(paths))
            for idx, (inp_path, opt_path1, opt_path2) in enumerate(paths):
                try:
                    if (
                        os.path.exists(opt_path1 + ".npy") == True
                        and os.path.exists(opt_path2 + ".npy") == True
                    ):
                        skipped += 1
                        continue
                    featur_pit = self.compute_f0(inp_path, f0_method)
                    if featur_pit is None:
                        skipped += 1
                        printt(i18n("音高全部为0，该音频无意义，跳过：%s") % inp_path)
                        continue
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
                    success += 1
                    if should_report(idx, len(paths), max_updates):
                        printt(
                            i18n("[F0提取] 进度：%s/%s | 成功：%s | 跳过：%s | %s")
                            % (idx + 1, len(paths), success, skipped, os.path.basename(inp_path))
                        )
                except Exception:
                    failed += 1
                    printt(
                        i18n("[F0提取][失败] %s\n%s")
                        % (inp_path, traceback.format_exc())
                    )
            printt(
                i18n("[F0提取] 完成 | 成功：%s | 跳过：%s | 失败：%s")
                % (success, skipped, failed)
            )


if __name__ == "__main__":
    # exp_dir=r"E:\codes\py39\dataset\mi-test"
    # n_p=16
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
        if os.path.exists(opt_path1 + ".npy") and os.path.exists(
            opt_path2 + ".npy"
        ):
            continue
        paths.append([inp_path, opt_path1, opt_path2])

    if mode == "cpu":
        if not paths:
            featureInput.go([], f0method, 1)
        else:
            worker_count = min(max(1, n_p), len(paths))
            ps = []
            for i in range(worker_count):
                p = Process(
                    target=featureInput.go,
                    args=(
                        paths[i::worker_count],
                        f0method,
                        max(1, (12 + worker_count - 1) // worker_count),
                    ),
                )
                ps.append(p)
                p.start()
            for p in ps:
                p.join()
    elif mode == "cuda":
        try:
            featureInput.go(
                paths[i_part::n_part],
                "rmvpe",
                max(1, (12 + n_part - 1) // n_part),
            )
        except Exception:
            printt(i18n("[F0提取][失败] %s") % traceback.format_exc())
    else:
        try:
            featureInput.go(paths, "rmvpe", 5)
        except Exception:
            printt(i18n("[F0提取][失败] %s") % traceback.format_exc())
