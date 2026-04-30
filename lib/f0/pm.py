from importlib import import_module
from typing import Protocol, cast

import numpy as np

from .f0 import F0Predictor, FilterRadius, FloatArray


class ParselmouthPitch(Protocol):
    selected_array: dict[str, FloatArray]


class ParselmouthSound(Protocol):
    def to_pitch_ac(
        self,
        *,
        time_step: float,
        voicing_threshold: float,
        pitch_floor: int,
        pitch_ceiling: int,
    ) -> ParselmouthPitch: ...


class ParselmouthModule(Protocol):
    def Sound(
        self, values: FloatArray, sampling_frequency: int
    ) -> ParselmouthSound: ...


parselmouth = cast(ParselmouthModule, import_module("parselmouth"))


class PM(F0Predictor):
    def __init__(self, hop_length=512, f0_min=50, f0_max=1100, sampling_rate=44100):
        super().__init__(hop_length, f0_min, f0_max, sampling_rate)

    def compute_f0(
        self,
        wav: FloatArray,
        p_len: int | None = None,
        filter_radius: FilterRadius = None,
    ) -> FloatArray:
        x = wav
        if p_len is None:
            p_len = x.shape[0] // self.hop_length
        else:
            assert abs(p_len - x.shape[0] // self.hop_length) < 4, "pad length error"
        time_step = self.hop_length / self.sampling_rate * 1000
        f0 = (
            parselmouth.Sound(x, self.sampling_rate)
            .to_pitch_ac(
                time_step=time_step / 1000,
                voicing_threshold=0.6,
                pitch_floor=self.f0_min,
                pitch_ceiling=self.f0_max,
            )
            .selected_array["frequency"]
        )

        pad_size = (p_len - len(f0) + 1) // 2
        if pad_size > 0 or p_len - len(f0) - pad_size > 0:
            f0 = np.asarray(
                np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"),
                dtype=np.float64,
            )
        return self._interpolate_f0(f0)[0]
