from math import log
from pathlib import Path
from typing import Optional, Union, Literal, Tuple

from numba import jit
import numpy as np
import torch


@jit(nopython=True)
def post_process(
    tf0: int,  # number of f0 points per second
    f0: np.ndarray,
    f0_up_key: int,
    manual_x_pad: int,
    f0_mel_min: float,
    f0_mel_max: float,
    manual_f0: Optional[Union[np.ndarray, list]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    f0 = np.multiply(f0, pow(2, f0_up_key / 12))
    # with open("test.txt","w")as f:f.write("\n".join([str(i)for i in f0.tolist()]))
    if manual_f0 is not None:
        delta_t = np.round(
            (manual_f0[:, 0].max() - manual_f0[:, 0].min()) * tf0 + 1
        ).astype("int16")
        replace_f0 = np.interp(
            list(range(delta_t)), manual_f0[:, 0] * 100, manual_f0[:, 1]
        )
        shape = f0[manual_x_pad * tf0 : manual_x_pad * tf0 + len(replace_f0)].shape[0]
        f0[manual_x_pad * tf0 : manual_x_pad * tf0 + len(replace_f0)] = replace_f0[
            :shape
        ]
    # with open("test_opt.txt","w")as f:f.write("\n".join([str(i)for i in f0.tolist()]))
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
        f0_mel_max - f0_mel_min
    ) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    f0_coarse = np.rint(f0_mel).astype(np.int32)
    return f0_coarse, f0  # 1-0


class Generator:
    def __init__(
        self,
        rmvpe_root: Path,
        is_half: bool,
        x_pad: int,
        device="cpu",
        window=160,
        sr=16000,
    ):
        self.rmvpe_root = rmvpe_root
        self.is_half = is_half
        self.x_pad = x_pad
        self.device = device
        self.window = window
        self.sr = sr

    def calculate(
        self,
        x: np.ndarray,
        p_len: Optional[int],
        f0_up_key: int,
        f0_method: Literal["pm", "dio", "harvest", "crepe", "rmvpe", "fcpe"],
        filter_radius: Optional[Union[int, float]],
        manual_f0: Optional[Union[np.ndarray, list]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if torch.is_tensor(x):
            x = x.cpu().numpy()
        f0_min = 50
        f0_max = 1100
        if f0_method == "pm":
            if not hasattr(self, "pm"):
                from .pm import PM

                self.pm = PM(self.window, f0_min, f0_max, self.sr)
            f0 = self.pm.compute_f0(x, p_len=p_len)
        elif f0_method == "dio":
            if not hasattr(self, "dio"):
                from .dio import Dio

                self.dio = Dio(self.window, f0_min, f0_max, self.sr)
            f0 = self.dio.compute_f0(x, p_len=p_len)
        elif f0_method == "harvest":
            if not hasattr(self, "harvest"):
                from .harvest import Harvest

                self.harvest = Harvest(self.window, f0_min, f0_max, self.sr)
            f0 = self.harvest.compute_f0(x, p_len=p_len, filter_radius=filter_radius)
        elif f0_method == "crepe":
            if not hasattr(self, "crepe"):
                from .crepe import CRePE

                self.crepe = CRePE(
                    self.window,
                    f0_min,
                    f0_max,
                    self.sr,
                    self.device,
                )
            f0 = self.crepe.compute_f0(x, p_len=p_len)
        elif f0_method == "rmvpe":
            if not hasattr(self, "rmvpe"):
                from .rmvpe import RMVPE

                self.rmvpe = RMVPE(
                    str(self.rmvpe_root / "rmvpe.pt"),
                    is_half=self.is_half,
                    device=self.device,
                    # use_jit=self.config.use_jit,
                )
            f0 = self.rmvpe.compute_f0(x, p_len=p_len, filter_radius=0.03)
            if "privateuseone" in str(self.device):  # clean ortruntime memory
                del self.rmvpe.model
                del self.rmvpe
        elif f0_method == "fcpe":
            if not hasattr(self, "fcpe"):
                from .fcpe import FCPE

                self.fcpe = FCPE(
                    self.window,
                    f0_min,
                    f0_max,
                    self.sr,
                    self.device,
                )
            f0 = self.fcpe.compute_f0(x, p_len=p_len)
        else:
            raise ValueError(f"f0 method {f0_method} has not yet been supported")

        return post_process(
            self.sr // self.window,
            f0,
            f0_up_key,
            self.x_pad,
            1127 * log(1 + f0_min / 700),
            1127 * log(1 + f0_max / 700),
            manual_f0,
        )
