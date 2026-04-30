from typing import Protocol, cast

import numpy as np
import pyworld
from numpy.typing import NDArray
from scipy import signal

from .f0 import F0Predictor, FilterRadius, FloatArray


class PyWorldHarvest(Protocol):
    def harvest(
        self,
        x: NDArray[np.float64],
        *,
        fs: int,
        f0_floor: int,
        f0_ceil: int,
        frame_period: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...

    def stonemask(
        self,
        x: NDArray[np.float64],
        f0: NDArray[np.float64],
        temporal_positions: NDArray[np.float64],
        fs: int,
    ) -> NDArray[np.float64]: ...


pyworld_harvest = cast(PyWorldHarvest, pyworld)


class Harvest(F0Predictor):
    def __init__(self, hop_length=512, f0_min=50, f0_max=1100, sampling_rate=44100):
        super().__init__(hop_length, f0_min, f0_max, sampling_rate)

    def compute_f0(
        self,
        wav: FloatArray,
        p_len: int | None = None,
        filter_radius: FilterRadius = None,
    ) -> FloatArray:
        if p_len is None:
            p_len = wav.shape[0] // self.hop_length
        f0, t = pyworld_harvest.harvest(
            wav.astype(np.double),
            fs=self.sampling_rate,
            f0_ceil=self.f0_max,
            f0_floor=self.f0_min,
            frame_period=1000 * self.hop_length / self.sampling_rate,
        )
        f0 = pyworld_harvest.stonemask(
            wav.astype(np.double), f0, t, self.sampling_rate
        )
        if filter_radius is not None and filter_radius > 2:
            f0 = signal.medfilt(f0, filter_radius)
        return self._interpolate_f0(self._resize_f0(f0, p_len))[0]
