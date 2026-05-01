from functools import lru_cache
from math import log
from pathlib import Path
from typing import Any, Literal, Protocol, cast
import hashlib
import warnings

from numba import jit
import numpy as np
from numpy.typing import NDArray
from scipy import signal
import torch
import torchcrepe

PitchMethod = Literal["pm", "harvest", "crepe", "rmvpe", "fcpe", "dio"]
PITCH_METHODS: list[PitchMethod] = ["pm", "harvest", "crepe", "rmvpe", "fcpe"]
ALL_PITCH_METHODS: tuple[PitchMethod, ...] = (
    "pm",
    "harvest",
    "crepe",
    "rmvpe",
    "fcpe",
    "dio",
)


class ParselmouthPitch(Protocol):
    selected_array: dict[str, NDArray[np.floating[Any]]]


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
        self, values: NDArray[np.floating[Any]], sampling_frequency: int
    ) -> ParselmouthSound: ...


class PyWorldModule(Protocol):
    def harvest(
        self,
        x: NDArray[np.float64],
        *,
        fs: int,
        f0_floor: int,
        f0_ceil: int,
        frame_period: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...

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


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API.*",
        category=UserWarning,
        module="pyworld",
    )
    import pyworld

import parselmouth

pyworld_api = cast(PyWorldModule, pyworld)
parselmouth_api = cast(ParselmouthModule, parselmouth)
_harvest_cache_audio: dict[str, NDArray[np.float64]] = {}


@jit(nopython=True)
def post_process(
    tf0: int,
    f0: np.ndarray,
    f0_up_key: int,
    manual_x_pad: int,
    f0_mel_min: float,
    f0_mel_max: float,
    manual_f0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    f0 = np.multiply(f0, pow(2, f0_up_key / 12))
    if manual_f0 is not None:
        delta_t = np.round(
            (manual_f0[:, 0].max() - manual_f0[:, 0].min()) * tf0 + 1
        ).astype("int16")
        replace_f0 = np.asarray(
            np.interp(np.arange(delta_t), manual_f0[:, 0] * 100, manual_f0[:, 1]),
            dtype=np.float64,
        )
        shape = f0[manual_x_pad * tf0 : manual_x_pad * tf0 + len(replace_f0)].shape[0]
        f0[manual_x_pad * tf0 : manual_x_pad * tf0 + len(replace_f0)] = replace_f0[
            :shape
        ]
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
        f0_mel_max - f0_mel_min
    ) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    f0_coarse = np.rint(f0_mel).astype(np.int32)
    return f0_coarse, f0


def _interpolate_f0(
    f0: NDArray[np.floating[Any]],
) -> tuple[NDArray[np.float64], NDArray[np.float32]]:
    data = np.asarray(f0, dtype=np.float64).reshape(-1)
    vuv_vector = np.zeros(data.shape[0], dtype=np.float32)
    vuv_vector[data > 0.0] = 1.0
    ip_data = data.copy()
    frame_number = data.shape[0]
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
            ip_data[i] = data[i]
            last_value = data[i]
    return np.asarray(ip_data, dtype=np.float64), vuv_vector


def _resize_f0(
    source: NDArray[np.floating[Any]], target_len: int
) -> NDArray[np.float64]:
    resized_source = np.array(source, dtype=np.float64)
    resized_source[resized_source < 0.001] = np.nan
    target = np.interp(
        np.arange(0, len(resized_source) * target_len, len(resized_source))
        / target_len,
        np.arange(0, len(resized_source)),
        resized_source,
    )
    return np.asarray(np.nan_to_num(target), dtype=np.float64)


def _hash_audio(audio: NDArray[np.float64]) -> str:
    return hashlib.sha1(audio.view(np.uint8)).hexdigest()


@lru_cache(maxsize=64)
def _cached_harvest(
    key: str, fs: int, f0_min: int, f0_max: int, frame_period: float
) -> NDArray[np.float64]:
    audio = _harvest_cache_audio[key]
    f0, t = pyworld_api.harvest(
        audio,
        fs=fs,
        f0_floor=f0_min,
        f0_ceil=f0_max,
        frame_period=frame_period,
    )
    return pyworld_api.stonemask(audio, f0, t, fs)


def extract_f0(
    x: np.ndarray,
    p_len: int | None,
    f0_up_key: int,
    f0_method: PitchMethod,
    filter_radius: int | float | None,
    *,
    rmvpe_root: Path,
    is_half: bool,
    x_pad: int,
    device: str,
    window: int,
    sr: int,
    state: dict[str, Any],
    manual_f0: np.ndarray | list[list[float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    audio = np.asarray(x, dtype=np.float64)
    if p_len is None:
        p_len = audio.shape[0] // window
    manual_f0_array = None if manual_f0 is None else np.asarray(manual_f0)
    f0_min = 50
    f0_max = 1100

    if f0_method == "pm":
        time_step = window / sr
        f0 = (
            parselmouth_api.Sound(audio, sr)
            .to_pitch_ac(
                time_step=time_step,
                voicing_threshold=0.6,
                pitch_floor=f0_min,
                pitch_ceiling=f0_max,
            )
            .selected_array["frequency"]
        )
        pad_size = (p_len - len(f0) + 1) // 2
        if pad_size > 0 or p_len - len(f0) - pad_size > 0:
            f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant")
        f0 = _interpolate_f0(np.asarray(f0, dtype=np.float64))[0]
    elif f0_method == "dio":
        dio_audio = audio.astype(np.double)
        f0, t = pyworld_api.dio(
            dio_audio,
            fs=sr,
            f0_floor=f0_min,
            f0_ceil=f0_max,
            frame_period=1000 * window / sr,
        )
        f0 = pyworld_api.stonemask(dio_audio, f0, t, sr)
        f0 = np.asarray([round(float(pitch), 1) for pitch in f0], dtype=np.float64)
        f0 = _interpolate_f0(_resize_f0(f0, p_len))[0]
    elif f0_method == "harvest":
        harvest_audio = audio.astype(np.double)
        cache_key = _hash_audio(harvest_audio)
        _harvest_cache_audio[cache_key] = harvest_audio
        f0 = _cached_harvest(cache_key, sr, f0_min, f0_max, 1000 * window / sr)
        if filter_radius is not None and filter_radius > 2:
            f0 = signal.medfilt(f0, int(filter_radius))
        f0 = _interpolate_f0(_resize_f0(f0, p_len))[0]
    elif f0_method == "crepe":
        audio_tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32))
        f0, pd = torchcrepe.predict(
            audio_tensor.float().to(device).unsqueeze(0),
            sr,
            window,
            f0_min,
            f0_max,
            batch_size=512,
            device=device,
            return_periodicity=True,
        )
        pd = torchcrepe.filter.median(pd, 3)
        f0 = torchcrepe.filter.mean(f0, 3)
        f0[pd < 0.1] = 0
        f0 = _interpolate_f0(
            _resize_f0(f0[0].detach().cpu().numpy(), p_len)
        )[0]
    elif f0_method == "rmvpe":
        if "rmvpe" not in state:
            from infer.lib.rmvpe import RMVPE

            state["rmvpe"] = RMVPE(
                str(rmvpe_root / "rmvpe.pt"),
                is_half=is_half,
                device=device,
            )
        rmvpe_audio = np.asarray(audio, dtype=np.float32)
        f0 = state["rmvpe"].infer_from_audio(rmvpe_audio, thred=0.03)
        f0 = _interpolate_f0(_resize_f0(np.asarray(f0, dtype=np.float64), p_len))[0]
    elif f0_method == "fcpe":
        if "fcpe" not in state:
            from torchfcpe import spawn_bundled_infer_model

            state["fcpe"] = spawn_bundled_infer_model(device)
        fcpe_audio = torch.from_numpy(np.asarray(audio, dtype=np.float32))
        threshold = float(filter_radius) if filter_radius is not None else 0.006
        f0 = (
            state["fcpe"]
            .infer(
                fcpe_audio.to(device).unsqueeze(0),
                sr=sr,
                decoder_mode="local_argmax",
                threshold=threshold,
            )
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )
        f0 = _interpolate_f0(_resize_f0(np.asarray(f0, dtype=np.float64), p_len))[0]
    else:
        raise ValueError(f"f0 method {f0_method} has not yet been supported")

    return post_process(
        sr // window,
        np.asarray(f0, dtype=np.float64),
        f0_up_key,
        x_pad,
        1127 * log(1 + f0_min / 700),
        1127 * log(1 + f0_max / 700),
        manual_f0_array,
    )


class Generator:
    def __init__(
        self,
        rmvpe_root: Path,
        is_half: bool,
        x_pad: int,
        device: str | torch.device | int = "cpu",
        window: int = 160,
        sr: int = 16000,
    ):
        self.rmvpe_root = rmvpe_root
        self.is_half = is_half
        self.x_pad = x_pad
        self.device = str(device)
        self.window = window
        self.sr = sr
        self._state: dict[str, Any] = {}

    def calculate(
        self,
        x: np.ndarray,
        p_len: int | None,
        f0_up_key: int,
        f0_method: PitchMethod,
        filter_radius: int | float | None,
        manual_f0: np.ndarray | list[list[float]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return extract_f0(
            x,
            p_len,
            f0_up_key,
            f0_method,
            filter_radius,
            rmvpe_root=self.rmvpe_root,
            is_half=self.is_half,
            x_pad=self.x_pad,
            device=self.device,
            window=self.window,
            sr=self.sr,
            state=self._state,
            manual_f0=manual_f0,
        )
