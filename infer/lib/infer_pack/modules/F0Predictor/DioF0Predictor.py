from typing import Protocol, cast

import numpy as np
import pyworld
from numpy.typing import NDArray

from infer.lib.infer_pack.modules import F0Predictor, FloatArray


class PyWorldDio(Protocol):
    def dio(
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


pyworld_dio = cast(PyWorldDio, pyworld)


class DioF0Predictor(F0Predictor):
    def __init__(
        self,
        hop_length: int = 512,
        f0_min: int = 50,
        f0_max: int = 1100,
        sampling_rate: int = 44100,
    ) -> None:
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.sampling_rate = sampling_rate

    def interpolate_f0(self, f0: FloatArray) -> tuple[FloatArray, FloatArray]:
        """
        对F0进行插值处理
        """

        data = np.reshape(f0, (f0.size, 1))

        vuv_vector = np.zeros((data.size, 1), dtype=np.float32)
        vuv_vector[data > 0.0] = 1.0
        vuv_vector[data <= 0.0] = 0.0

        ip_data = data

        frame_number = data.size
        last_value = 0.0
        for i in range(frame_number):
            if data[i] <= 0.0:
                j = i + 1
                for j in range(i + 1, frame_number):
                    if data[j] > 0.0:
                        break
                if j < frame_number - 1:
                    if last_value > 0.0:
                        step = (data[j] - data[i - 1]) / float(j - i)
                        for k in range(i, j):
                            ip_data[k] = data[i - 1] + step * (k - i + 1)
                    else:
                        for k in range(i, j):
                            ip_data[k] = data[j]
                else:
                    for k in range(i, frame_number):
                        ip_data[k] = last_value
            else:
                ip_data[i] = data[i]  # 这里可能存在一个没有必要的拷贝
                last_value = data[i]

        return ip_data[:, 0], vuv_vector[:, 0]

    def resize_f0(self, x: FloatArray, target_len: int) -> FloatArray:
        source = np.array(x)
        source[source < 0.001] = np.nan
        target = np.interp(
            np.arange(0, len(source) * target_len, len(source)) / target_len,
            np.arange(0, len(source)),
            source,
        )
        res = np.asarray(np.nan_to_num(target), dtype=np.float64)
        return res

    def compute_f0(self, wav: FloatArray, p_len: int | None = None) -> FloatArray:
        if p_len is None:
            p_len = wav.shape[0] // self.hop_length
        f0, t = pyworld_dio.dio(
            wav.astype(np.double),
            fs=self.sampling_rate,
            f0_floor=self.f0_min,
            f0_ceil=self.f0_max,
            frame_period=1000 * self.hop_length / self.sampling_rate,
        )
        f0 = pyworld_dio.stonemask(wav.astype(np.double), f0, t, self.sampling_rate)
        for index, pitch in enumerate(f0):
            f0[index] = round(pitch, 1)
        return self.interpolate_f0(self.resize_f0(f0, p_len))[0]

    def compute_f0_uv(
        self, wav: FloatArray, p_len: int | None = None
    ) -> tuple[FloatArray, FloatArray]:
        if p_len is None:
            p_len = wav.shape[0] // self.hop_length
        f0, t = pyworld_dio.dio(
            wav.astype(np.double),
            fs=self.sampling_rate,
            f0_floor=self.f0_min,
            f0_ceil=self.f0_max,
            frame_period=1000 * self.hop_length / self.sampling_rate,
        )
        f0 = pyworld_dio.stonemask(wav.astype(np.double), f0, t, self.sampling_rate)
        for index, pitch in enumerate(f0):
            f0[index] = round(pitch, 1)
        return self.interpolate_f0(self.resize_f0(f0, p_len))
