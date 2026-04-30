from abc import ABC, abstractmethod
import numpy as np
import parselmouth
import torch
import torchcrepe
from scipy import signal
from typing import Protocol, cast
import warnings
from functools import lru_cache
import hashlib

# Use a context manager to suppress the warning during import
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API.*",
        category=UserWarning,
        module="pyworld",  # Optional, but adds precision
    )
    import pyworld


class PyWorldModule(Protocol):
    def harvest(
        self,
        x: np.ndarray,
        fs: int,
        f0_ceil: int,
        f0_floor: int,
        frame_period: float,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def stonemask(
        self, x: np.ndarray, f0: np.ndarray, t: np.ndarray, fs: int
    ) -> np.ndarray: ...


pyworld_api = cast(PyWorldModule, pyworld)


# --- Base Pitch Extractor Class ---
class PitchExtractor(ABC):
    """Abstract base class for all pitch extraction methods."""

    def __init__(self, sample_rate: int, window_size: int, f0_min: int, f0_max: int):
        self.sr = sample_rate
        self.window = window_size
        self.f0_min = f0_min
        self.f0_max = f0_max

    @abstractmethod
    def extract_pitch(self, audio: np.ndarray, p_len: int) -> np.ndarray:
        """Extracts and returns the fundamental frequency (f0) from audio."""
        pass


# --- Concrete Pitch Extractor Classes with Lazy Loading ---
class PM_PitchExtractor(PitchExtractor):
    def extract_pitch(self, audio: np.ndarray, p_len: int) -> np.ndarray:
        time_step = self.window / self.sr * 1000
        f0 = (
            parselmouth.Sound(audio, self.sr)
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
            f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant")
        return f0


input_audio_path2wav = {}

_wav_cache = {}  # holds np.ndarray -> key mapping


def _hash_array(arr: np.ndarray) -> str:
    """Hash an ndarray to use as cache key."""
    return hashlib.sha1(arr.view(np.uint8)).hexdigest()


@lru_cache(maxsize=64)
def cache_harvest_f0_cached(
    key: str, fs: int, f0max: int, f0min: int, frame_period: float
):
    # the actual waveform is stored in a global dict keyed by hash
    audio = _wav_cache[key]
    f0, t = pyworld_api.harvest(
        audio,
        fs=fs,
        f0_ceil=f0max,
        f0_floor=f0min,
        frame_period=frame_period,
    )
    f0 = pyworld_api.stonemask(audio, f0, t, fs)
    return f0


class Harvest_PitchExtractor(PitchExtractor):
    def __init__(
        self,
        sample_rate: int,
        window_size: int,
        f0_min: int,
        f0_max: int,
        filter_radius: int = 3,
    ):
        super().__init__(sample_rate, window_size, f0_min, f0_max)
        self.filter_radius = filter_radius
        # No model to load, logic is self-contained.

    def extract_pitch(self, audio: np.ndarray, p_len: int) -> np.ndarray:
        # Assuming cache_harvest_f0_cached and _hash_array are defined elsewhere
        key = _hash_array(audio.astype(np.double))
        _wav_cache[key] = audio.astype(np.double)
        f0 = cache_harvest_f0_cached(key, self.sr, self.f0_max, self.f0_min, 10)
        if self.filter_radius > 2:
            f0 = signal.medfilt(f0, 3)
        return f0


class Crepe_PitchExtractor(PitchExtractor):
    def __init__(
        self, sample_rate: int, window_size: int, f0_min: int, f0_max: int, device: str
    ):
        super().__init__(sample_rate, window_size, f0_min, f0_max)
        self.device = device
        # The torchcrepe model is loaded implicitly by the `predict` function.

    def extract_pitch(self, audio: np.ndarray, p_len: int) -> np.ndarray:
        model = "full"
        batch_size = 512
        audio_tensor = torch.tensor(np.copy(audio))[None].float()
        f0, pd = torchcrepe.predict(
            audio_tensor,
            self.sr,
            self.window,
            self.f0_min,
            self.f0_max,
            model,
            batch_size=batch_size,
            device=self.device,
            return_periodicity=True,
        )
        pd = torchcrepe.filter.median(pd, 3)
        f0 = torchcrepe.filter.mean(f0, 3)
        f0[pd < 0.1] = 0
        return f0[0].cpu().numpy()


class RMVPE_PitchExtractor(PitchExtractor):
    def __init__(
        self,
        sample_rate: int,
        window_size: int,
        f0_min: int,
        f0_max: int,
        device: str,
        is_half: bool,
        # shared,
    ):
        import shared

        super().__init__(sample_rate, window_size, f0_min, f0_max)
        self.device = device
        self.is_half = is_half
        self.shared = shared
        self.model = None  # Initialize model as None

    def extract_pitch(self, audio: np.ndarray, p_len: int) -> np.ndarray:
        if self.model is None:
            # Lazy loading: The model is only loaded the first time this method is called.
            from infer.lib.rmvpe import RMVPE

            self.model = RMVPE(
                "%s/rmvpe.pt" % (self.shared.rmvpe_root),
                is_half=self.is_half,
                device=self.device,
            )
        f0 = self.model.infer_from_audio(audio, thred=0.03)
        if "privateuseone" in str(self.device) and self.model:
            del self.model.model
            del self.model
            self.model = None
        return f0


# class Pipeline:
#     def __init__(self, sr: int, window, is_half, device):
#         self.sr = sr
#         self.window = window
#         self.is_half = is_half
#         self.device = device
#         self.f0_min = 50
#         self.f0_max = 1100
#         # Initialize an empty dictionary for pitch extractors.
#         self.pitch_extractors = {}
#         # Assuming other pipeline initialization here...

#     def get_f0(
#         self,
#         x: np.ndarray,
#         p_len: int,
#         f0_up_key: int,
#         f0_method: str,  # Use str for type hinting
#         filter_radius: int = 3,
#         inp_f0: Optional[np.ndarray] = None,
#     ):
#         # Check if the required pitch extractor instance exists, otherwise create it.
#         if f0_method not in self.pitch_extractors:
#             if f0_method == "pm":
#                 self.pitch_extractors[f0_method] = PM_PitchExtractor(
#                     self.sr, self.window, self.f0_min, self.f0_max
#                 )
#             elif f0_method == "harvest":
#                 self.pitch_extractors[f0_method] = Harvest_PitchExtractor(
#                     self.sr, self.window, self.f0_min, self.f0_max, filter_radius
#                 )
#             elif f0_method == "crepe":
#                 self.pitch_extractors[f0_method] = Crepe_PitchExtractor(
#                     self.sr, self.window, self.f0_min, self.f0_max, self.device
#                 )
#             elif f0_method == "rmvpe":
#                 self.pitch_extractors[f0_method] = RMVPE_PitchExtractor(
#                     self.sr,
#                     self.window,
#                     self.f0_min,
#                     self.f0_max,
#                     self.device,
#                     self.is_half,
#                     # shared,
#                 )

#         # Get the correct extractor and call the method.
#         f0 = self.pitch_extractors[f0_method].extract_pitch(x, p_len)
#         f0 *= pow(2, f0_up_key / 12)
#         tf0 = self.sr // self.window
#         if inp_f0 is not None:
#             delta_t = np.round(
#                 (inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1
#             ).astype("int16")
#             replace_f0 = np.interp(
#                 list(range(delta_t)), inp_f0[:, 0] * 100, inp_f0[:, 1]
#             )
#             shape = f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)].shape[0]
#             f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)] = replace_f0[
#                 :shape
#             ]

#         f0bak = f0.copy()
#         f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
#         f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
#         f0_mel = 1127 * np.log(1 + f0 / 700)
#         f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
#             f0_mel_max - f0_mel_min
#         ) + 1
#         f0_mel[f0_mel <= 1] = 1
#         f0_mel[f0_mel > 255] = 255
#         f0_coarse = np.rint(f0_mel).astype(np.int32)
#         return f0_coarse, f0bak
