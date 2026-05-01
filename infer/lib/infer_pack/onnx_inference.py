import librosa
import numpy as np
import onnxruntime
import soundfile
from pathlib import Path
from typing import Literal

import logging

from lib.f0 import extract_f0

logger = logging.getLogger(__name__)


class ContentVec:
    def __init__(self, vec_path="pretrained/vec-768-layer-12.onnx", device=None):
        logger.info("Load model(s) from {}".format(vec_path))
        if device == "cpu" or device is None:
            providers = ["CPUExecutionProvider"]
        elif device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            raise RuntimeError("Unsportted Device")
        self.model = onnxruntime.InferenceSession(vec_path, providers=providers)

    def __call__(self, wav):
        return self.forward(wav)

    def forward(self, wav):
        feats = wav
        if feats.ndim == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.ndim == 1, feats.ndim
        feats = np.expand_dims(np.expand_dims(feats, 0), 0)
        onnx_input = {self.model.get_inputs()[0].name: feats}
        logits = np.asarray(self.model.run(None, onnx_input)[0], dtype=np.float32)
        return logits.transpose(0, 2, 1)


class OnnxRVC:
    def __init__(
        self,
        model_path,
        sr=40000,
        hop_size=512,
        vec_path="vec-768-layer-12",
        device="cpu",
    ):
        vec_path = f"pretrained/{vec_path}.onnx"
        self.vec_model = ContentVec(vec_path, device)
        if device == "cpu" or device is None:
            providers = ["CPUExecutionProvider"]
        elif device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            raise RuntimeError("Unsportted Device")
        self.model = onnxruntime.InferenceSession(model_path, providers=providers)
        self.sampling_rate = sr
        self.hop_size = hop_size

    def forward(self, hubert, hubert_length, pitch, pitchf, ds, rnd):
        onnx_input = {
            self.model.get_inputs()[0].name: hubert,
            self.model.get_inputs()[1].name: hubert_length,
            self.model.get_inputs()[2].name: pitch,
            self.model.get_inputs()[3].name: pitchf,
            self.model.get_inputs()[4].name: ds,
            self.model.get_inputs()[5].name: rnd,
        }
        output = np.asarray(self.model.run(None, onnx_input)[0], dtype=np.float32)
        return (output * 32767).astype(np.int16)

    def inference(
        self,
        raw_path,
        sid,
        f0_method: Literal["pm", "harvest", "dio"] = "dio",
        f0_up_key=0,
        pad_time=0.5,
        cr_threshold=0.02,
    ):
        wav, sr = librosa.load(raw_path, sr=self.sampling_rate)
        org_length = len(wav)
        if org_length / sr > 50.0:
            raise RuntimeError("Reached Max Length")

        wav16k = librosa.resample(wav, orig_sr=self.sampling_rate, target_sr=16000)
        wav16k = wav16k

        hubert = self.vec_model(wav16k)
        hubert = np.repeat(hubert, 2, axis=2).transpose(0, 2, 1).astype(np.float32)
        hubert_length = hubert.shape[1]

        pitch, pitchf = extract_f0(
            wav,
            hubert_length,
            int(f0_up_key),
            f0_method,
            cr_threshold,
            rmvpe_root=Path("assets/rmvpe"),
            is_half=False,
            x_pad=0,
            device="cpu",
            window=self.hop_size,
            sr=self.sampling_rate,
            state={},
        )
        pitch = pitch.astype(np.int64)

        pitchf = pitchf.reshape(1, len(pitchf)).astype(np.float32)
        pitch = pitch.reshape(1, len(pitch))
        ds = np.array([sid]).astype(np.int64)

        rnd = np.random.randn(1, 192, hubert_length).astype(np.float32)
        hubert_length = np.array([hubert_length]).astype(np.int64)

        out_wav = self.forward(hubert, hubert_length, pitch, pitchf, ds, rnd).squeeze()
        out_wav = np.pad(out_wav, (0, 2 * self.hop_size), "constant")
        return out_wav[0:org_length]
