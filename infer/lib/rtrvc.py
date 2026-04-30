from io import BytesIO
import os
from typing import Union, Literal, Optional
from pathlib import Path

import fairseq
import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import Resample

from lib.f0 import Generator
from lib.synthesizer import load_synthesizer
from lib.types import FileLike


class RVC:
    def __init__(
        self,
        key: Union[int, float],
        formant: Union[int, float],
        pth_path: FileLike,  # type: ignore
        index_path: str,
        index_rate: Union[int, float],
        n_cpu: int = os.cpu_count(),
        device: Union[str, torch.device, int] = "cpu",
        use_jit: bool = False,
        is_half: bool = False,
        is_dml: bool = False,
    ) -> None:
        if is_dml:

            def forward_dml(ctx, x, scale):
                ctx.scale = scale
                res = x.clone().detach()
                return res

            fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml

        self.device = device
        self.f0_up_key = key
        self.formant_shift = formant
        self.sr = 16000  # hubert sampling rate
        self.window = 160  # hop length
        self.f0_min = 50
        self.f0_max = 1100
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.n_cpu = n_cpu
        self.use_jit = use_jit
        self.is_half = is_half

        if index_rate > 0:
            self.index = faiss.read_index(index_path)
            self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)

        self.pth_path = pth_path
        self.index_path = index_path
        self.index_rate = index_rate

        self.cache_pitch: torch.Tensor = torch.zeros(
            1024, device=self.device, dtype=torch.long
        )
        self.cache_pitchf = torch.zeros(1024, device=self.device, dtype=torch.float32)

        self.resample_kernel = {}

        self.f0_gen = Generator(
            Path(os.environ["rmvpe_root"]), is_half, 0, device, self.window, self.sr
        )

        models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            ["assets/hubert/hubert_base.pt"],
            suffix="",
        )
        hubert_model = models[0]
        hubert_model = hubert_model.to(self.device)
        if self.is_half:
            try:
                hubert_model = hubert_model.half()
            except Exception as e:
                print(
                    "Warning: could not convert HuBERT to half — keeping float32. Error:",
                    e,
                )
                hubert_model = hubert_model.float()
        else:
            hubert_model = hubert_model.float()
        hubert_model.eval()
        self.hubert = hubert_model

        self.net_g: Optional[nn.Module] = None

        def set_default_model():
            self.net_g, cpt = load_synthesizer(self.pth_path, self.device)
            self.tgt_sr = cpt["config"][-1]
            cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
            self.if_f0 = cpt.get("f0", 1)
            self.version = cpt.get("version", "v1")
            if self.is_half:
                try:
                    self.net_g = self.net_g.half()
                except Exception as e:
                    # fallback: keep float32 on GPU (or move to cpu if GPU unusable)
                    print(
                        "Warning: could not convert model to half — keeping float32. Error:",
                        e,
                    )
            else:
                self.net_g = self.net_g.float()

        def set_jit_model():
            from lib.jit import get_jit_model
            from lib.synthesizer import synthesizer_jit_export

            cpt = get_jit_model(self.pth_path, self.is_half, synthesizer_jit_export)

            self.tgt_sr = cpt["config"][-1]
            self.if_f0 = cpt.get("f0", 1)
            self.version = cpt.get("version", "v1")
            self.net_g = torch.jit.load(BytesIO(cpt["model"]), map_location=self.device)
            self.net_g.infer = self.net_g.forward
            self.net_g.eval().to(self.device)

        if (
            self.use_jit
            and not is_dml
            and not (self.is_half and "cpu" in str(self.device))
        ):
            set_jit_model()
        else:
            set_default_model()

    def set_key(self, new_key):
        self.f0_up_key = new_key

    def set_formant(self, new_formant):
        self.formant_shift = new_formant

    def set_index_rate(self, new_index_rate):
        if new_index_rate > 0 and self.index_rate <= 0:
            self.index = faiss.read_index(self.index_path)
            self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
        self.index_rate = new_index_rate

    def infer(
        self,
        input_wav: torch.Tensor,
        block_frame_16k: int,
        skip_head: int,
        return_length: int,
        f0method: Union[tuple, str],
        protect: float = 1.0,
    ) -> np.ndarray:
        with torch.no_grad():
            if self.is_half:
                feats = input_wav.half()
            else:
                feats = input_wav.float()
            feats = feats.to(self.device)
            if feats.dim() == 2:  # double channels
                feats = feats.mean(-1)
            feats = feats.view(1, -1)
            padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)

            inputs = {
                "source": feats,
                "padding_mask": padding_mask,
                "output_layer": 9 if self.version == "v1" else 12,
            }
            logits = self.hubert.extract_features(**inputs)
            feats = (
                self.hubert.final_proj(logits[0]) if self.version == "v1" else logits[0]
            )
            feats = torch.cat((feats, feats[:, -1:, :]), 1)
            if protect < 0.5 and self.if_f0 == 1:
                feats0 = feats.clone()

        try:
            if hasattr(self, "index") and self.index_rate > 0:
                npy = feats[0][skip_head // 2 :].cpu().numpy()
                if self.is_half:
                    npy = npy.astype("float32")
                score, ix = self.index.search(npy, k=8)
                if (ix >= 0).all():
                    weight = np.square(1 / score)
                    weight /= weight.sum(axis=1, keepdims=True)
                    npy = np.sum(
                        self.big_npy[ix] * np.expand_dims(weight, axis=2), axis=1
                    )
                    if self.is_half:
                        npy = npy.astype("float16")
                    feats[0][skip_head // 2 :] = (
                        torch.from_numpy(npy).unsqueeze(0).to(self.device)
                        * self.index_rate
                        + (1 - self.index_rate) * feats[0][skip_head // 2 :]
                    )
        except:
            pass

        p_len = input_wav.shape[0] // self.window
        factor = pow(2, self.formant_shift / 12)
        return_length2 = int(np.ceil(return_length * factor))
        cache_pitch = cache_pitchf = None
        pitch = pitchf = None
        if isinstance(f0method, tuple):
            pitch, pitchf = f0method
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()
        elif self.if_f0 == 1:
            f0_extractor_frame = block_frame_16k + 800
            if f0method == "rmvpe":
                f0_extractor_frame = (
                    5120 * ((f0_extractor_frame - 1) // 5120 + 1) - self.window
                )
            pitch, pitchf = self._get_f0(
                input_wav[-f0_extractor_frame:],
                self.f0_up_key - self.formant_shift,
                method=f0method,
            )
            shift = block_frame_16k // self.window
            self.cache_pitch[:-shift] = self.cache_pitch[shift:].clone()
            self.cache_pitchf[:-shift] = self.cache_pitchf[shift:].clone()
            self.cache_pitch[4 - pitch.shape[0] :] = pitch[3:-1]
            self.cache_pitchf[4 - pitch.shape[0] :] = pitchf[3:-1]
            cache_pitch = self.cache_pitch[None, -p_len:]
            cache_pitchf = (
                self.cache_pitchf[None, -p_len:] * return_length2 / return_length
            )

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        feats = feats[:, :p_len, :]
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )
            feats0 = feats0[:, :p_len, :]
            pitchff = pitchf.clone()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = protect
            pitchff = pitchff.unsqueeze(-1)
            feats = feats * pitchff + feats0 * (1 - pitchff)
            feats = feats.to(feats0.dtype)
        p_len = torch.LongTensor([p_len]).to(self.device)
        sid = torch.LongTensor([0]).to(self.device)
        with torch.no_grad():
            infered_audio = (
                self.net_g.infer(
                    feats,
                    p_len,
                    sid,
                    pitch=cache_pitch,
                    pitchf=cache_pitchf,
                    skip_head=skip_head,
                    return_length=return_length,
                    return_length2=return_length2,
                )
                .squeeze(1)
                .float()
            )
        upp_res = int(np.floor(factor * self.tgt_sr // 100))
        if upp_res != self.tgt_sr // 100:
            if upp_res not in self.resample_kernel:
                self.resample_kernel[upp_res] = Resample(
                    orig_freq=upp_res,
                    new_freq=self.tgt_sr // 100,
                    dtype=torch.float32,
                ).to(self.device)
            infered_audio = self.resample_kernel[upp_res](
                infered_audio[:, : return_length * upp_res]
            )
        return infered_audio.squeeze()

    def _get_f0(
        self,
        x: torch.Tensor,
        f0_up_key: Union[int, float],
        filter_radius: Optional[Union[int, float]] = None,
        method: Literal["crepe", "rmvpe", "fcpe", "pm", "harvest", "dio"] = "fcpe",
    ):
        c, f = self.f0_gen.calculate(x, None, f0_up_key, method, filter_radius)
        if not torch.is_tensor(c):
            c = torch.from_numpy(c)
        if not torch.is_tensor(f):
            f = torch.from_numpy(f)
        return c.long().to(self.device), f.float().to(self.device)
