import os
from pathlib import Path
from typing import Literal, cast
from os import PathLike

import librosa
import numpy as np
import onnxruntime

from lib.f0 import Generator, PitchMethod

type ModelPath = str | bytes | PathLike[str]


class Model:
    def __init__(
        self,
        path: ModelPath,
        device: Literal["cpu", "cuda"] = "cpu",
    ):
        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            raise RuntimeError("Unsportted Device")
        self.model = onnxruntime.InferenceSession(path, providers=providers)


class ContentVec(Model):
    def __init__(
        self,
        vec_path: ModelPath,
        device: Literal["cpu", "cuda"] = "cpu",
    ):
        super().__init__(vec_path, device)

    def __call__(self, wav: np.ndarray):
        return self.forward(wav)

    def forward(self, wav: np.ndarray) -> np.ndarray:
        if wav.ndim == 2:  # double channels
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        wav = np.expand_dims(np.expand_dims(wav, 0), 0)
        onnx_input = {self.model.get_inputs()[0].name: wav}
        logits = cast(np.ndarray, self.model.run(None, onnx_input)[0])
        return logits.transpose(0, 2, 1)


class RVC(Model):
    def __init__(
        self,
        model_path: ModelPath,
        hop_len: int = 512,
        model_sr: int = 40000,
        vec_path: ModelPath = "vec-768-layer-12.onnx",
        device: Literal["cpu", "cuda"] = "cpu",
    ):
        super().__init__(model_path, device)
        self.vec_model = ContentVec(vec_path, device)
        self.hop_len = hop_len
        self.f0_gen = Generator(Path("assets/rmvpe"), False, 0, window=hop_len, sr=model_sr)

    def infer(
        self,
        wav: np.ndarray,
        wav_sr: int,
        sid: int = 0,
        f0_method: PitchMethod = "dio",
        f0_up_key: int = 0,
    ) -> np.ndarray:
        org_length = len(wav)
        if org_length / wav_sr > 50.0:
            raise RuntimeError("wav max length exceeded")

        hubert = self.vec_model(librosa.resample(wav, orig_sr=wav_sr, target_sr=16000))
        hubert = np.repeat(hubert, 2, axis=2).transpose(0, 2, 1).astype(np.float32)
        hubert_length = hubert.shape[1]

        pitch, pitchf = self.f0_gen.calculate(
            wav, hubert_length, f0_up_key, f0_method, None
        )
        pitch = pitch.astype(np.int64)

        pitchf = pitchf.reshape(1, len(pitchf)).astype(np.float32)
        pitch = pitch.reshape(1, len(pitch))
        ds = np.array([sid]).astype(np.int64)

        rnd = np.random.randn(1, 192, hubert_length).astype(np.float32)
        hubert_length_array = np.array([hubert_length]).astype(np.int64)

        out_wav = self.forward(hubert, hubert_length_array, pitch, pitchf, ds, rnd).squeeze()

        out_wav = np.pad(out_wav, (0, 2 * self.hop_len), "constant")

        return out_wav[0:org_length]

    def forward(
        self,
        hubert: np.ndarray,
        hubert_length: np.ndarray,
        pitch: np.ndarray,
        pitchf: np.ndarray,
        ds: np.ndarray,
        rnd: np.ndarray,
    ) -> np.ndarray:
        onnx_input = {
            self.model.get_inputs()[0].name: hubert,
            self.model.get_inputs()[1].name: hubert_length,
            self.model.get_inputs()[2].name: pitch,
            self.model.get_inputs()[3].name: pitchf,
            self.model.get_inputs()[4].name: ds,
            self.model.get_inputs()[5].name: rnd,
        }
        audio = cast(np.ndarray, self.model.run(None, onnx_input)[0])
        return (audio * 32767).astype(np.int16)
