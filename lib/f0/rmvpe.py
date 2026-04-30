from io import BytesIO
import os
from pathlib import Path
from typing import Literal, Protocol, cast

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from lib.jit.jit import load_inputs, get_jit_model, export_jit_model, save_pickle

from .mel import MelSpectrogram
from .f0 import F0Predictor, FilterRadius, FloatArray
from .models import get_rmvpe


class OnnxInput(Protocol):
    name: str


class OnnxModel(Protocol):
    def get_inputs(self) -> list[OnnxInput]: ...

    def get_outputs(self) -> list[OnnxInput]: ...

    def run(
        self, output_names: list[str], input_feed: dict[str, NDArray[np.floating]]
    ) -> list[FloatArray]: ...


def rmvpe_jit_export(
    model_path: str | Path,
    mode: Literal["script", "trace"] = "script",
    inputs_path: str | Path | None = None,
    save_path: str | Path | None = None,
    device: str = "cpu",
    is_half: bool = False,
):
    model_path = Path(model_path)
    if not save_path:
        stem = model_path.with_suffix("")
        save_path = stem.with_suffix(".half.jit" if is_half else ".jit")
    else:
        save_path = Path(save_path)
    if "cuda" in str(device) and ":" not in str(device):
        device = "cuda:0"

    model = get_rmvpe(str(model_path), device, is_half)
    inputs: dict[str, torch.Tensor] = {}
    if mode == "trace":
        if inputs_path is None:
            raise ValueError("inputs_path is required when mode is 'trace'")
        inputs = cast(dict[str, torch.Tensor], load_inputs(inputs_path, device, is_half))
    ckpt = export_jit_model(model, mode, inputs, device, is_half)
    ckpt["device"] = str(device)
    save_pickle(ckpt, save_path)
    return ckpt


class RMVPE(F0Predictor):
    def __init__(
        self,
        model_path: str,
        is_half: bool,
        device: str,
        use_jit=False,
    ):
        hop_length = 160
        f0_min = 30
        f0_max = 8000
        sampling_rate = 16000

        super().__init__(
            hop_length,
            f0_min,
            f0_max,
            sampling_rate,
            device,
        )

        self.is_half = is_half
        self.model: OnnxModel | torch.nn.Module
        cents_mapping = 20 * np.arange(360) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4))  # 368

        self.mel_extractor = MelSpectrogram(
            is_half=is_half,
            n_mel_channels=128,
            sampling_rate=sampling_rate,
            win_length=1024,
            hop_length=hop_length,
            mel_fmin=f0_min,
            mel_fmax=f0_max,
            device=self.device,
        ).to(self.device)

        def rmvpe_jit_model():
            ckpt = get_jit_model(Path(model_path), is_half, self.device, rmvpe_jit_export)
            model = torch.jit.load(BytesIO(ckpt["model"]), map_location=self.device)
            model = model.to(self.device)
            return model

        if use_jit and not (is_half and "cpu" in str(self.device)):
            self.model = rmvpe_jit_model()
        else:
            self.model = get_rmvpe(model_path, self.device, is_half)

    def compute_f0(
        self,
        wav: FloatArray,
        p_len: int | None = None,
        filter_radius: FilterRadius = None,
    ) -> FloatArray:
        if p_len is None:
            p_len = wav.shape[0] // self.hop_length
        wav_tensor = torch.from_numpy(wav)
        mel = self.mel_extractor(
            wav_tensor.float().to(self.device).unsqueeze(0), center=True
        )
        hidden = self._mel2hidden(mel)
        if isinstance(hidden, torch.Tensor):
            hidden = hidden.squeeze(0).cpu().numpy()
        else:
            hidden = hidden[0]
        if self.is_half == True:
            hidden = hidden.astype("float32")

        f0 = self._decode(hidden, thred=filter_radius)

        return self._interpolate_f0(self._resize_f0(f0, p_len))[0]

    def _to_local_average_cents(self, salience, threshold=0.05):
        center = np.argmax(salience, axis=1)  # Frame length #index
        salience = np.pad(salience, ((0, 0), (4, 4)))  # Frame length, 368
        center += 4
        todo_salience = []
        todo_cents_mapping = []
        starts = center - 4
        ends = center + 5
        for idx in range(salience.shape[0]):
            todo_salience.append(salience[:, starts[idx] : ends[idx]][idx])
            todo_cents_mapping.append(self.cents_mapping[starts[idx] : ends[idx]])
        todo_salience = np.array(todo_salience)  # Frame length, 9
        todo_cents_mapping = np.array(todo_cents_mapping)  # Frame length, 9
        product_sum = np.sum(todo_salience * todo_cents_mapping, 1)
        weight_sum = np.sum(todo_salience, 1)  # Frame length
        devided = product_sum / weight_sum  # Frame length
        maxx = np.max(salience, axis=1)  # Frame length
        devided[maxx <= threshold] = 0
        return devided

    def _mel2hidden(self, mel: torch.Tensor) -> torch.Tensor | FloatArray:
        with torch.no_grad():
            n_frames = mel.shape[-1]
            n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
            if n_pad > 0:
                mel = F.pad(mel, (0, n_pad), mode="constant")
            mel = mel.half() if self.is_half else mel.float()
            torch_model = cast(torch.nn.Module, self.model)
            hidden = torch_model(mel)
            return hidden[:, :n_frames]

    def _decode(self, hidden, thred=0.03):
        if thred is None:
            thred = 0.03
        cents_pred = self._to_local_average_cents(hidden, threshold=thred)
        f0 = 10 * (2 ** (cents_pred / 1200))
        f0[f0 == 10] = 0
        # f0 = np.array([10 * (2 ** (cent_pred / 1200)) if cent_pred else 0 for cent_pred in cents_pred])
        return f0
