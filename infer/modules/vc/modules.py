from lib.f0 import PitchMethod
import traceback
import logging
from pathlib import Path
from typing import Any, Optional, Union
import gradio as gr
import resampy

from configs.config import Config

logger = logging.getLogger(__name__)
import numpy as np
import torch
from infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from infer.modules.vc.pipeline import Pipeline
from infer.modules.vc.utils import *


def resample_audio(
    audio_array: np.ndarray,
    orig_sr: int,
    target_sr: int,
):
    # Check if the audio is stereo and downmix to mono
    if audio_array.ndim > 1 and audio_array.shape[1] > 1:
        # print("Detected stereo audio, downmixing to mono.")
        # Average the channels to create a mono signal
        audio_mono = audio_array.mean(axis=1)
    else:
        # Already mono or 1D array
        audio_mono = audio_array.flatten()  # Ensure it's 1D in case it's (N, 1)

    # print(f"Mono audio shape after downmixing: {audio_mono.shape}")

    if audio_mono.size < 10:  # A reasonable minimum length for resampling
        raise ValueError(
            f"Mono audio signal length ({audio_mono.size}) is too small to resample from {orig_sr} to {target_sr}. "
            "Ensure the audio file contains actual sound data."
        )

    # Perform resampling on the mono signal
    resampled_audio = resampy.resample(audio_mono, orig_sr, target_sr)
    # print(f"Resampled audio shape: {resampled_audio.shape}")
    return resampled_audio


class VC:
    def __init__(self: "VC", config: Config):
        # self.config = config
        self.n_spk: Optional[int] = None
        self.tgt_sr: Optional[int] = None
        self.net_g: Optional[
            Union[
                SynthesizerTrnMs256NSFsid,
                SynthesizerTrnMs256NSFsid_nono,
                SynthesizerTrnMs768NSFsid,
                SynthesizerTrnMs768NSFsid_nono,
            ]
        ] = None
        self.pipeline: Optional[Pipeline] = None
        self.cpt: dict[str, Any] | None = None
        self.version: str = "UNKNOWN"
        self.if_f0: Optional[int] = None
        self.hubert_model: Optional[HubertModel] = None
        self.config: Config = config

        # ## Real-time inference state ##
        self.audio_buffer = np.array([], dtype=np.float32)
        # Pitch cache for continuity between chunks
        self.cache_pitch: torch.Tensor = torch.zeros(1, 256, dtype=torch.long).to(
            self.config.device
        )
        self.cache_pitchf: torch.Tensor = torch.zeros(1, 256, dtype=torch.float32).to(
            self.config.device
        )

    def get_vc(self: "VC", sid: Optional[str], *to_return_protect):
        if sid is None or sid == "":
            logger.warning("No SID")
            return (
                {"visible": True, "value": 0.5, "__type__": "update"},
                {"choices": [], "value": "", "__type__": "update"},
            )
        # self.pipeline
        logger.info(f"Get sid: {sid}")

        to_return_protect0 = {
            "visible": self.if_f0 != 0,
            "value": (
                to_return_protect[0] if self.if_f0 != 0 and to_return_protect else 0.5
            ),
            "__type__": "update",
        }

        if sid == "" or sid == []:
            if self.hubert_model is not None:
                # Considering polling, we need to add a check to see if sid switched from having a model to not having one
                logger.info("Clean model cache")
                self.hubert_model = self.net_g = self.n_spk = self.hubert_model = (
                    self.tgt_sr
                ) = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # You just have to follow this instruction to clear it.
                cpt = self.cpt
                if cpt is not None:
                    self.if_f0 = cpt.get("f0", 1)
                    self.version = cpt.get("version", "v1")
                    if self.version == "v1":
                        if self.if_f0 == 1:
                            self.net_g = SynthesizerTrnMs256NSFsid(
                                *cpt["config"], is_half=self.config.is_half
                            )
                        else:
                            self.net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
                    elif self.version == "v2":
                        if self.if_f0 == 1:
                            self.net_g = SynthesizerTrnMs768NSFsid(
                                *cpt["config"], is_half=self.config.is_half
                            )
                        else:
                            self.net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
                self.net_g = None
                self.cpt = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return (
                {
                    "visible": True,
                    "value": to_return_protect0,
                    "__type__": "update",
                },
                "",
                "",
            )
        person = shared.weight_root / sid
        logger.info(f"Loading: {person}")

        self.cpt = torch.load(person, map_location="cpu", weights_only=False)
        self.tgt_sr = self.cpt["config"][-1]
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        self.if_f0 = self.cpt.get("f0", 1)
        self.version = self.cpt.get("version", "v1")

        synthesizer_class = {
            ("v1", 1): SynthesizerTrnMs256NSFsid,
            ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
            ("v2", 1): SynthesizerTrnMs768NSFsid,
            ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
        }

        self.net_g = synthesizer_class.get(
            (self.version, self.if_f0), SynthesizerTrnMs256NSFsid
        )(*self.cpt["config"], is_half=self.config.is_half)

        del self.net_g.enc_q

        self.net_g.load_state_dict(self.cpt["weight"], strict=False)
        self.net_g.eval().to(self.config.device)
        if self.config.is_half:
            try:
                self.net_g = self.net_g.half()
            except Exception as e:
                self.net_g = self.net_g.float()
                print(
                    "Warning: could not convert model to half — keeping float32. Error:",
                    e,
                )
        else:
            self.net_g = self.net_g.float()

        self.pipeline = Pipeline(self.tgt_sr, self.config)
        # n_spk = self.cpt["config"][-3]
        index = {"value": get_index_path_from_model(sid), "__type__": "update"}
        logger.info(f"Select index: {index['value']}")
        res = (
            (
                to_return_protect0,
                index,
            )
            # if to_return_protect
            # else {"visible": True, "maximum": n_spk, "__type__": "update"}
        )
        logger.info(f"Result {res}")

        return res

    def vc_single(
        self: "VC",
        sr_and_audio: Optional[Tuple[int, np.ndarray]],
        f0_up_key: int,
        f0_method: PitchMethod,
        file_index: Optional[str],  # Path to .index file from dropdown
        index_rate: float,
        resample_sr: int,  # Target sample rate
        rms_mix_rate: float,
        protect: float,
        progress: gr.Progress = gr.Progress(),
    ) -> tuple[str, tuple[int, np.ndarray] | None]:
        if self.net_g is None or self.pipeline is None:
            return "Model not loaded. Please select a valid SID.", None
        file_index = None
        f0_file = None
        sid = 0
        filter_radius = 3
        if_f0 = self.if_f0
        if if_f0 is None:
            return "Model F0 setting unknown. Please reload the model.", None
        tgt_sr = self.tgt_sr
        if tgt_sr is None:
            return "Model target sample rate unknown. Please reload the model.", None
        f0_up_key = int(f0_up_key)
        try:
            if sr_and_audio is None:
                return "Audio is required", None

            original_sr, audio = sr_and_audio
            if original_sr != 16000:
                # print(f"Resampling audio from {original_sr} Hz to {16000} Hz")
                audio = resample_audio(audio, original_sr, 16000)
            audio_max: np.float64 = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max
            times = [0.0, 0.0, 0.0]
            if self.hubert_model is None:
                self.hubert_model = load_hubert(self.config)
            if file_index is None:
                file_index = ""

            audio_opt: np.ndarray = self.pipeline.pipeline(
                model=self.hubert_model,
                net_g=self.net_g,
                sid=sid,
                audio=audio,
                # input_audio_path="NA",
                times=times,
                f0_up_key=f0_up_key,
                f0_method=f0_method,
                file_index=file_index,
                index_rate=index_rate,
                if_f0=if_f0,
                # filter_radius=filter_radius,
                tgt_sr=tgt_sr,
                resample_sr=resample_sr,
                rms_mix_rate=rms_mix_rate,
                version=self.version,
                protect=protect,
                f0_file=f0_file,
                progress=progress,
            )
            if self.tgt_sr != resample_sr >= 16000:
                tgt_sr = resample_sr
            index_info = (
                f"Using Index: \n{file_index}"
                if file_index and Path(file_index).exists()
                else "Index not used."
            )
            return (
                f"Success.\n{index_info}\nTime:\nnpy: {times[0]:.2f}s, f0: {times[1]:.2f}s, infer: {times[2]:.2f}s.",
                (tgt_sr, audio_opt),
            )
        except:
            info = traceback.format_exc()
            logger.warning(info)
            return f"Failed with error:\n{info}", None

    def vc_realtime(
        self: "VC",
        sr_and_audio: tuple[int, np.ndarray] | None,
        f0_up_key: int,
        f0_method: PitchMethod,
        file_index: Optional[str],  # Path to .index file from dropdown
        index_rate: float,
        resample_sr: int,  # Target sample rate
        rms_mix_rate: float,
        protect: float,
        block_size: int = 5120,  # default chunk size in samples at 16kHz (~320ms)
        crossfade_size: int = 512,  # overlap for smoother transitions
    ) -> tuple[int, np.ndarray] | None:
        """
        Real-time voice conversion with buffering and pitch caching.
        Reuses vc_single() internally to avoid code duplication.
        """
        if sr_and_audio is None:
            return None

        sr, audio_chunk = sr_and_audio
        if sr != 16000:
            audio_chunk = resample_audio(audio_chunk, sr, 16000)

        # Append new audio to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])

        # Process only when buffer has enough samples
        if len(self.audio_buffer) < block_size:
            return None  # wait for more audio

        # Take a block with padding for continuity
        process_block = self.audio_buffer[:block_size].copy()

        # Keep a tail for continuity to next round
        self.audio_buffer = self.audio_buffer[block_size - crossfade_size :]

        # Call existing single VC function
        msg, result = self.vc_single(
            (16000, process_block),
            f0_up_key=f0_up_key,
            f0_method=f0_method,
            file_index=file_index,
            index_rate=index_rate,
            resample_sr=resample_sr,
            rms_mix_rate=rms_mix_rate,
            protect=protect,
        )

        if result is None:
            return None

        tgt_sr, audio_opt = result

        # Apply crossfade smoothing at boundaries
        if hasattr(self, "_prev_tail") and self._prev_tail is not None:
            fade_len = min(crossfade_size, len(audio_opt), len(self._prev_tail))
            fade_in = np.linspace(0, 1, fade_len)
            fade_out = 1 - fade_in
            audio_opt[:fade_len] = (
                self._prev_tail[-fade_len:] * fade_out + audio_opt[:fade_len] * fade_in
            )

        # Save tail for next round
        self._prev_tail = audio_opt

        return tgt_sr, audio_opt
