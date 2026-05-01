from io import BytesIO
import os
import pickle
import sys
import traceback
from pathlib import Path
from typing import Protocol, cast
from infer.lib import jit
from infer.lib.jit.get_synthesizer import get_synthesizer
from time import time as ttime
import fairseq
import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.data.dictionary import Dictionary
from torch.serialization import safe_globals

from infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from lib.f0 import ALL_PITCH_METHODS, Generator, PitchMethod

now_dir = os.getcwd()
sys.path.append(now_dir)

from configs.config import Config

# config = Config()


class InferModule(Protocol):
    def infer(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]: ...


def printt(strr, *args):
    if len(args) == 0:
        print(strr)
    else:
        print(strr % args)


# config.device=torch.device("cpu")######## Force cpu testing
# config.is_half=False######## Force cpu testing
class RVC:
    def __init__(
        self,
        key,
        pth_path,
        index_path,
        index_rate,
        n_cpu,
        inp_q,
        opt_q,
        config: Config,
        last_rvc=None,
    ) -> None:
        """
        Initialize
        """
        try:
            # global config
            self.config = config
            self.inp_q = inp_q
            self.opt_q = opt_q
            # device="cpu"######## Force cpu testing
            self.device: str | torch.device = config.device
            self.f0_up_key = key
            self.n_cpu = n_cpu
            self.use_jit = self.config.use_jit
            self.is_half = config.is_half

            if index_rate != 0:
                self.index = faiss.read_index(index_path)
                self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
                printt("Index search enabled")
            self.pth_path: str = pth_path
            self.index_path = index_path
            self.index_rate = index_rate
            self.cache_pitch: torch.Tensor = torch.zeros(
                1024, device=self.device, dtype=torch.long
            )
            self.cache_pitchf = torch.zeros(
                1024, device=self.device, dtype=torch.float32
            )
            self.f0_gen = Generator(
                Path("assets/rmvpe"),
                self.is_half,
                0,
                self.device,
                160,
                16000,
            )

            if last_rvc is None:
                with safe_globals([Dictionary]):
                    models, _, _ = (
                        fairseq.checkpoint_utils.load_model_ensemble_and_task(
                            ["assets/hubert/hubert_base.pt"],
                            suffix="",
                        )
                    )
                hubert_model = models[0]
                hubert_model = hubert_model.to(self.device)
                if self.is_half:
                    hubert_model = hubert_model.half()
                else:
                    hubert_model = hubert_model.float()
                hubert_model.eval()
                self.model = hubert_model
            else:
                self.model = last_rvc.model

            self.net_g: nn.Module | None = None

            def set_default_model():
                self.net_g, cpt = get_synthesizer(self.pth_path, self.device)
                self.tgt_sr = cpt["config"][-1]
                cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
                self.if_f0 = cpt.get("f0", 1)
                self.version = cpt.get("version", "v1")
                if self.is_half:
                    self.net_g = self.net_g.half()
                else:
                    self.net_g = self.net_g.float()

            def set_jit_model():
                jit_pth_path = self.pth_path.rstrip(".pth")
                jit_pth_path += ".half.jit" if self.is_half else ".jit"
                reload = False
                cpt = None
                if str(self.device) == "cuda":
                    self.device = torch.device("cuda:0")
                if os.path.exists(jit_pth_path):
                    cpt = jit.load(jit_pth_path)
                    model_device = cpt["device"]
                    if model_device != str(self.device):
                        reload = True
                else:
                    reload = True

                if reload:
                    cpt = jit.synthesizer_jit_export(
                        self.pth_path,
                        "script",
                        None,
                        device=self.device,
                        is_half=self.is_half,
                    )

                if cpt is None:
                    raise RuntimeError("JIT checkpoint failed to load")
                self.tgt_sr = cpt["config"][-1]
                self.if_f0 = cpt.get("f0", 1)
                self.version = cpt.get("version", "v1")
                self.net_g = torch.jit.load(
                    BytesIO(cpt["model"]), map_location=self.device
                )
                self.net_g.infer = self.net_g.forward
                self.net_g.eval().to(self.device)

            def set_synthesizer():
                if self.use_jit:
                    if self.is_half and "cpu" in str(self.device):
                        printt(
                            "Use default Synthesizer model. \
                                    Jit is not supported on the CPU for half floating point"
                        )
                        set_default_model()
                    else:
                        set_jit_model()
                else:
                    set_default_model()

            if last_rvc is None or last_rvc.pth_path != self.pth_path:
                set_synthesizer()
            else:
                self.tgt_sr = last_rvc.tgt_sr
                self.if_f0 = last_rvc.if_f0
                self.version = last_rvc.version
                self.is_half = last_rvc.is_half
                if last_rvc.use_jit != self.use_jit:
                    set_synthesizer()
                else:
                    self.net_g = last_rvc.net_g
            if self.net_g is None:
                raise RuntimeError("RVC model failed to load")

            if last_rvc is not None and hasattr(last_rvc, "model_rmvpe"):
                self.model_rmvpe = last_rvc.model_rmvpe
            if last_rvc is not None and hasattr(last_rvc, "model_fcpe"):
                self.device_fcpe = last_rvc.device_fcpe
                self.model_fcpe = last_rvc.model_fcpe
            if last_rvc is not None and hasattr(last_rvc, "f0_gen"):
                self.f0_gen = last_rvc.f0_gen
        except:
            printt(traceback.format_exc())

    def change_key(self, new_key):
        self.f0_up_key = new_key

    def change_index_rate(self, new_index_rate):
        if new_index_rate != 0 and self.index_rate == 0:
            self.index = faiss.read_index(self.index_path)
            self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
            printt("Index search enabled")
        self.index_rate = new_index_rate

    def _get_f0(
        self,
        x: torch.Tensor,
        f0_up_key: int,
        method: PitchMethod,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coarse, fine = self.f0_gen.calculate(
            x.detach().cpu().numpy(),
            None,
            f0_up_key,
            method,
            3,
        )
        return (
            torch.from_numpy(coarse).long().to(self.device),
            torch.from_numpy(fine).float().to(self.device),
        )

    def infer(
        self,
        input_wav: torch.Tensor,
        block_frame_16k,
        skip_head,
        return_length,
        f0method,
    ) -> torch.Tensor:
        t1 = ttime()
        with torch.no_grad():
            if self.config.is_half:
                feats = input_wav.half().view(1, -1)
            else:
                feats = input_wav.float().view(1, -1)
            padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)
            inputs = {
                "source": feats,
                "padding_mask": padding_mask,
                "output_layer": 9 if self.version == "v1" else 12,
            }
            logits = self.model.extract_features(**inputs)
            feats = (
                self.model.final_proj(logits[0]) if self.version == "v1" else logits[0]
            )
            feats = torch.cat((feats, feats[:, -1:, :]), 1)
        t2 = ttime()
        try:
            if hasattr(self, "index") and self.index_rate != 0:
                npy = feats[0][skip_head // 2 :].cpu().numpy().astype("float32")
                score, ix = self.index.search(npy, k=8)
                if (ix >= 0).all():
                    weight = np.square(1 / score)
                    weight /= weight.sum(axis=1, keepdims=True)
                    npy = np.sum(
                        self.big_npy[ix] * np.expand_dims(weight, axis=2), axis=1
                    )
                    if self.config.is_half:
                        npy = npy.astype("float16")
                    feats[0][skip_head // 2 :] = (
                        torch.from_numpy(npy).unsqueeze(0).to(self.device)
                        * self.index_rate
                        + (1 - self.index_rate) * feats[0][skip_head // 2 :]
                    )
                else:
                    printt(
                        "Invalid index. You MUST use added_xxxx.index but not trained_xxxx.index!"
                    )
            else:
                printt("Index search FAILED or disabled")
        except:
            traceback.print_exc()
            printt("Index search FAILED")
        t3 = ttime()
        p_len = input_wav.shape[0] // 160
        cache_pitch: torch.Tensor | None = None
        cache_pitchf: torch.Tensor | None = None
        if self.if_f0 == 1:
            if f0method not in ALL_PITCH_METHODS:
                raise ValueError(f"Unsupported f0 method: {f0method}")
            method = cast(PitchMethod, f0method)
            f0_extractor_frame = block_frame_16k + 800
            if method == "rmvpe":
                f0_extractor_frame = 5120 * ((f0_extractor_frame - 1) // 5120 + 1) - 160
            pitch, pitchf = self._get_f0(
                input_wav[-f0_extractor_frame:],
                self.f0_up_key,
                method,
            )
            shift = block_frame_16k // 160
            self.cache_pitch[:-shift] = self.cache_pitch[shift:].clone()
            self.cache_pitchf[:-shift] = self.cache_pitchf[shift:].clone()
            self.cache_pitch[4 - pitch.shape[0] :] = pitch[3:-1]
            self.cache_pitchf[4 - pitch.shape[0] :] = pitchf[3:-1]
            cache_pitch = self.cache_pitch[None, -p_len:]
            cache_pitchf = self.cache_pitchf[None, -p_len:]
        t4 = ttime()
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        feats = feats[:, :p_len, :]
        p_len = torch.LongTensor([p_len]).to(self.device)
        sid = torch.LongTensor([0]).to(self.device)
        skip_head = torch.LongTensor([skip_head])
        return_length = torch.LongTensor([return_length])
        net_g = cast(InferModule, self.net_g)
        with torch.no_grad():
            if self.if_f0 == 1:
                if cache_pitch is None or cache_pitchf is None:
                    raise RuntimeError("Pitch cache was not initialized")
                infered_audio, _, _ = net_g.infer(
                    feats,
                    p_len,
                    cache_pitch,
                    cache_pitchf,
                    sid,
                    skip_head,
                    return_length,
                )
            else:
                infered_audio, _, _ = net_g.infer(
                    feats, p_len, sid, skip_head, return_length
                )
        t5 = ttime()
        printt(
            "Spent time: fea = %.3fs, index = %.3fs, f0 = %.3fs, model = %.3fs",
            t2 - t1,
            t3 - t2,
            t4 - t3,
            t5 - t4,
        )
        return infered_audio.squeeze().float()
