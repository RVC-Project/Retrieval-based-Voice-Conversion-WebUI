from numpy.typing import NDArray
import numpy as np


class Pitch:
    t1: float
    selected_array: dict[str, NDArray[np.floating]]


class Sound:
    def __init__(
        self, values: NDArray[np.floating], sampling_frequency: int
    ) -> None: ...

    def to_pitch_ac(
        self,
        time_step: float,
        voicing_threshold: float,
        pitch_floor: int | float,
        pitch_ceiling: int | float,
    ) -> Pitch: ...
