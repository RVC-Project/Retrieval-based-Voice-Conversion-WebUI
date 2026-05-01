import os
import sys
import traceback
import logging
from pathlib import Path
from typing import Protocol, TypeAlias

from configs.config import Config
from infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)

import gradio as gr

logger = logging.getLogger(__name__)

from time import time as ttime
from fairseq.models.hubert.hubert import (
    HubertModel as FairseqHubertModel,
)  # Renamed for clarity in this example
import faiss
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal

from lib.f0 import Generator, PitchMethod

now_dir = os.getcwd()
sys.path.append(now_dir)

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)  # type: ignore


class NamedFile(Protocol):
    name: str


RVCModel: TypeAlias = (
    SynthesizerTrnMs256NSFsid
    | SynthesizerTrnMs256NSFsid_nono
    | SynthesizerTrnMs768NSFsid
    | SynthesizerTrnMs768NSFsid_nono
)


def change_rms(
    data1: np.ndarray, sr1: int, data2: np.ndarray, sr2: int, rate: float
):  # 1 is input audio, 2 is output audio, rate is the proportion of 2
    # print(data1.max(),data2.max())
    rms1 = librosa.feature.rms(
        y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2
    )  # One point every half second
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)
    rms1 = torch.from_numpy(rms1)
    rms1 = F.interpolate(
        rms1.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.from_numpy(rms2)
    rms2 = F.interpolate(
        rms2.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
    data2 *= (
        torch.pow(rms1, torch.tensor(1 - rate))
        * torch.pow(rms2, torch.tensor(rate - 1))
    ).numpy()
    return data2


class Pipeline:
    import shared

    sr: int = 16000
    window: int = 160

    f0_min = 50
    f0_max = 1100

    def __init__(self, tgt_sr: int, config: Config) -> None:
        self.x_pad, self.x_query, self.x_center, self.x_max, self.is_half = (
            config.x_pad,
            config.x_query,
            config.x_center,
            config.x_max,
            config.is_half,
        )
        self.t_pad: int = self.sr * self.x_pad  # Pad time around each entry
        self.t_pad_tgt: int = tgt_sr * self.x_pad
        self.t_pad2: int = self.t_pad * 2
        self.t_query: int = self.sr * self.x_query  # Query time around the query point
        self.t_center: int = self.sr * self.x_center  # Query point position
        self.t_max: int = self.sr * self.x_max  # Threshold for skipping queries
        self.device: str = config.device
        self.f0_gen = Generator(
            self.shared.rmvpe_root,
            self.is_half,
            self.x_pad,
            self.device,
            self.window,
            self.sr,
        )

    def get_f0(
        self,
        x: np.ndarray,
        p_len: int,
        f0_up_key: int,
        f0_method: PitchMethod,
        filter_radius: int = 3,
        inp_f0: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.f0_gen.calculate(
            x,
            p_len,
            f0_up_key,
            f0_method,
            filter_radius,
            inp_f0,
        )

    def vc(
        self,
        model: FairseqHubertModel,
        net_g: RVCModel,
        sid: torch.Tensor,
        audio: np.ndarray,
        pitch: torch.Tensor | None,
        pitchf: torch.Tensor | None,
        times: list[float],
        index: faiss.Index | None,
        big_npy: np.ndarray | None,
        index_rate: float,
        version: str,
        protect: float,
    ) -> np.ndarray:  # ,file_index,file_big_npy
        feats = torch.from_numpy(audio)
        if self.is_half:
            try:
                feats = feats.half()
            except Exception as e:
                print(
                    "Warning: could not convert audio features to half — keeping float32. Error:",
                    e,
                )
                feats = feats.float()
        else:
            feats = feats.float()
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)

        inputs = {
            "source": feats.to(self.device),
            "padding_mask": padding_mask,
            "output_layer": 9 if version == "v1" else 12,
        }
        t0 = ttime()
        with torch.no_grad():
            logits = model.extract_features(**inputs)
            feats = model.final_proj(logits[0]) if version == "v1" else logits[0]
        feats0: torch.Tensor | None = None
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = feats.clone()
            #    not isinstance(index, type(None))
            # and not isinstance(big_npy, type(None))
            # and index_rate != 0

        if index is not None and big_npy is not None and index_rate != 0:
            npy = feats[0].cpu().numpy()
            if self.is_half:
                npy = npy.astype("float32")

            score, ix = index.search(npy, k=8)  # type: ignore[missing-argument]
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)

            if self.is_half:
                npy = npy.astype("float16")
            feats = (
                torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
                + (1 - index_rate) * feats
            )

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if protect < 0.5 and pitch is not None and pitchf is not None:
            assert feats0 is not None
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )
        t1 = ttime()
        p_len = audio.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch is not None and pitchf is not None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]

        if protect < 0.5 and pitch is not None and pitchf is not None:
            assert feats0 is not None
            pitchff = pitchf.clone()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = protect
            pitchff = pitchff.unsqueeze(-1)
            feats = feats * pitchff + feats0 * (1 - pitchff)
            feats = feats.to(feats0.dtype)
        p_len = torch.tensor([p_len], device=self.device).long()
        with torch.no_grad():
            hasp = pitch is not None and pitchf is not None
            arg = (feats, p_len, pitch, pitchf, sid) if hasp else (feats, p_len, sid)
            audio1: np.ndarray = (net_g.infer(*arg)[0][0, 0]).data.cpu().float().numpy()
            del hasp, arg
        del feats, p_len, padding_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t2 = ttime()
        times[0] += t1 - t0
        times[2] += t2 - t1
        return audio1

    def pipeline(
        self,
        model: FairseqHubertModel,
        net_g: RVCModel,
        sid: int,
        audio: np.ndarray,
        # input_audio_path: str,
        times: list[float],
        f0_up_key: int,
        f0_method: PitchMethod,
        file_index: str,
        index_rate: float,
        if_f0: int,
        # filter_radius: int,
        tgt_sr: int,
        resample_sr: int,
        rms_mix_rate: float,
        version: str,
        protect: float,
        f0_file: NamedFile | None = None,
        progress=gr.Progress(),
    ) -> np.ndarray:
        progress(0.01, desc="Initializing...")  # Initial progress

        if (
            file_index != ""
            # and file_big_npy != ""
            # and os.path.exists(file_big_npy) == True
            and Path(file_index).exists()
            and index_rate != 0
        ):
            try:
                loaded_index = faiss.read_index(file_index)
                index: faiss.Index | None = loaded_index
                # big_npy = np.load(file_big_npy)
                big_npy: np.ndarray | None = loaded_index.reconstruct_n(  # type: ignore[missing-argument]
                    0, loaded_index.ntotal
                )
            except:
                traceback.print_exc()
                index = None
                big_npy = None
        else:
            index = None
            big_npy = None
        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += np.abs(audio_pad[i : i - self.window])
            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t
                    - self.t_query
                    + np.where(
                        audio_sum[t - self.t_query : t + self.t_query]
                        == audio_sum[t - self.t_query : t + self.t_query].min()
                    )[0][0]
                )
        s = 0
        audio_segments: list[np.ndarray] = []
        t = None
        t1 = ttime()
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        inp_f0: np.ndarray | None = None
        if f0_file is not None:
            try:
                with open(f0_file.name, "r") as f:
                    lines = f.read().strip("\n").split("\n")
                inp_f0_rows: list[list[float]] = []
                for line in lines:
                    inp_f0_rows.append([float(i) for i in line.split(",")])
                inp_f0 = np.array(inp_f0_rows, dtype="float32")
            except:
                traceback.print_exc()
        sid_tensor = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        pitch: torch.Tensor | None = None
        pitchf: torch.Tensor | None = None
        if if_f0 == 1:
            progress(0.2, desc="Extracting F0...")  # Progress update
            pitch_np, pitchf_np = self.get_f0(
                # input_audio_path,
                x=audio_pad,
                p_len=p_len,
                f0_up_key=f0_up_key,
                f0_method=f0_method,
                # filter_radius=filter_radius,
                inp_f0=inp_f0,
            )
            pitch_np = pitch_np[:p_len]
            pitchf_np = pitchf_np[:p_len]
            if "mps" not in str(self.device) or "xpu" not in str(self.device):
                pitchf_np = pitchf_np.astype(np.float32)
            pitch = torch.tensor(pitch_np, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf_np, device=self.device).unsqueeze(0).float()
        t2 = ttime()
        times[1] += t2 - t1

        total_segments = len(opt_ts) + 1  # +1 for the last segment
        # for t in opt_ts:
        for i, t in enumerate(opt_ts):
            progress(
                (i / total_segments) * 0.7 + 0.25,
                desc=f"Converting segment {i+1}/{total_segments}...",
            )  # Progress update
            t = t // self.window * self.window
            if if_f0 == 1:
                assert pitch is not None
                assert pitchf is not None
                audio_segments.append(
                    self.vc(
                        model,
                        net_g,
                        sid_tensor,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        pitch[:, s // self.window : (t + self.t_pad2) // self.window],
                        pitchf[:, s // self.window : (t + self.t_pad2) // self.window],
                        times,
                        index,
                        big_npy,
                        index_rate,
                        version,
                        protect,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            else:
                audio_segments.append(
                    self.vc(
                        model,
                        net_g,
                        sid_tensor,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        None,
                        None,
                        times,
                        index,
                        big_npy,
                        index_rate,
                        version,
                        protect,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            s = t

        progress(
            0.95, desc="Finalizing conversion..."
        )  # Progress update before last segment

        if if_f0 == 1:
            assert pitch is not None
            assert pitchf is not None
            audio_segments.append(
                self.vc(
                    model,
                    net_g,
                    sid_tensor,
                    audio_pad[t:],
                    pitch[:, t // self.window :] if t is not None else pitch,
                    pitchf[:, t // self.window :] if t is not None else pitchf,
                    times,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        else:
            audio_segments.append(
                self.vc(
                    model=model,
                    net_g=net_g,
                    sid=sid_tensor,
                    audio=audio_pad[t:],
                    pitch=None,
                    pitchf=None,
                    times=times,
                    index=index,
                    big_npy=big_npy,
                    index_rate=index_rate,
                    version=version,
                    protect=protect,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        audio_opt = np.concatenate(audio_segments)
        if rms_mix_rate != 1:
            audio_opt = change_rms(audio, 16000, audio_opt, tgt_sr, rms_mix_rate)
        if tgt_sr != resample_sr >= 16000:
            audio_opt = librosa.resample(
                audio_opt, orig_sr=tgt_sr, target_sr=resample_sr
            )
        audio_max = np.abs(audio_opt).max() / 0.99
        max_int16 = 32768
        if audio_max > 1:
            max_int16 /= audio_max
        audio_opt = (audio_opt * max_int16).astype(np.int16)
        del pitch, pitchf, sid_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        progress(1.0, desc="Conversion complete.")  # Final progress

        return audio_opt
