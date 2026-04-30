from importlib import import_module
from typing import Protocol, cast

import numpy as np

from infer.lib.infer_pack.modules import F0Predictor, FloatArray


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


class PMF0Predictor(F0Predictor):
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
        Perform interpolation on F0
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
                ip_data[i] = data[i]  # There might be an unnecessary copy here
                last_value = data[i]

        return ip_data[:, 0], vuv_vector[:, 0]

    def compute_f0(self, wav: FloatArray, p_len: int | None = None) -> FloatArray:
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
        f0, uv = self.interpolate_f0(f0)
        return f0

    def compute_f0_uv(
        self, wav: FloatArray, p_len: int | None = None
    ) -> tuple[FloatArray, FloatArray]:
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
        f0, uv = self.interpolate_f0(f0)
        return f0, uv
