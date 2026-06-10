"""UI-agnostic realtime voice conversion engine, extracted from gui_v1.py.

Owns the sounddevice full-duplex stream, the SOLA / noise-gate / buffer
pipeline, and the infer.lib.rtrvc.RVC model. Any front-end (Electron,
tests) drives it through start() / stop() / set_param().
"""

import os
import sys
import time

import librosa
import numpy as np
import sounddevice as sd
import torch
import torch.nn.functional as F
import torchaudio.transforms as tat

now_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if now_dir not in sys.path:
    sys.path.append(now_dir)

from configs.config import Config
from infer.lib import rtrvc as rvc_for_realtime
from tools.torchgate import TorchGate


def printt(strr, *args):
    if len(args) == 0:
        print(strr)
    else:
        print(strr % args)


def phase_vocoder(a, b, fade_out, fade_in):
    window = torch.sqrt(fade_out * fade_in)
    fa = torch.fft.rfft(a * window)
    fb = torch.fft.rfft(b * window)
    absab = torch.abs(fa) + torch.abs(fb)
    n = a.shape[0]
    if n % 2 == 0:
        absab[1:-1] *= 2
    else:
        absab[1:] *= 2
    phia = torch.angle(fa)
    phib = torch.angle(fb)
    deltaphase = phib - phia
    deltaphase = deltaphase - 2 * np.pi * torch.floor(deltaphase / 2 / np.pi + 0.5)
    w = 2 * np.pi * torch.arange(n // 2 + 1).to(a) + deltaphase
    t = torch.arange(n).unsqueeze(-1).to(a) / n
    result = (
        a * (fade_out**2)
        + b * (fade_in**2)
        + torch.sum(absab * torch.cos(w * t + phia), -1) * window / n
    )
    return result


class EngineConfig:
    def __init__(self) -> None:
        self.pth_path: str = ""
        self.index_path: str = ""
        self.pitch: int = 0
        self.formant: float = 0.0
        self.sr_type: str = "sr_model"
        self.block_time: float = 0.25  # s
        self.threhold: int = -60
        self.crossfade_time: float = 0.05
        self.extra_time: float = 2.5
        self.I_noise_reduce: bool = False
        self.O_noise_reduce: bool = False
        self.use_pv: bool = False
        self.rms_mix_rate: float = 0.0
        self.index_rate: float = 0.0
        self.n_cpu: int = 4
        self.f0method: str = "rmvpe"
        self.sg_hostapi: str = ""
        self.sg_wasapi_exclusive: bool = False
        self.sg_input_device: str = ""
        self.sg_output_device: str = ""
        self.samplerate: int = 0
        self.channels: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "EngineConfig":
        cfg = cls()
        for key, value in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        # gui_v1's config file stores crossfade under "crossfade_length"
        if "crossfade_length" in data:
            cfg.crossfade_time = data["crossfade_length"]
        return cfg


class RealtimeVC:
    """Realtime voice conversion engine. No UI imports.

    status_callback is called with a dict (e.g. {"type": "stats",
    "infer_time_ms": 42}) from the PortAudio realtime callback thread,
    so it must return without blocking (e.g. hand off with
    loop.call_soon_threadsafe or Queue.put_nowait, never blocking I/O).
    """

    def __init__(self, inp_q, opt_q, status_callback=None):
        self.config = Config()
        self.config.use_jit = False
        self.cfg = EngineConfig()
        self.inp_q = inp_q
        self.opt_q = opt_q
        self.status_callback = status_callback
        self.function = "vc"
        self.delay_time = 0.0
        self.stream = None
        self.flag_vc = False
        self.rvc = None
        self.hostapis = None
        self.input_devices = None
        self.output_devices = None
        self.input_devices_indices = None
        self.output_devices_indices = None
        self.update_devices()

    # ---- device handling (verbatim from gui_v1.GUI) ----

    def update_devices(self, hostapi_name=None):
        self.flag_vc = False
        sd._terminate()
        sd._initialize()
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        for hostapi in hostapis:
            for device_idx in hostapi["devices"]:
                devices[device_idx]["hostapi_name"] = hostapi["name"]
        self.hostapis = [hostapi["name"] for hostapi in hostapis]
        if hostapi_name not in self.hostapis:
            hostapi_name = self.hostapis[0]
        self.input_devices = [
            d["name"]
            for d in devices
            if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
        ]
        self.output_devices = [
            d["name"]
            for d in devices
            if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name
        ]
        self.input_devices_indices = [
            d["index"] if "index" in d else d["name"]
            for d in devices
            if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
        ]
        self.output_devices_indices = [
            d["index"] if "index" in d else d["name"]
            for d in devices
            if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name
        ]

    def set_devices(self, input_device, output_device):
        sd.default.device[0] = self.input_devices_indices[
            self.input_devices.index(input_device)
        ]
        sd.default.device[1] = self.output_devices_indices[
            self.output_devices.index(output_device)
        ]
        printt("Input device: %s:%s", str(sd.default.device[0]), input_device)
        printt("Output device: %s:%s", str(sd.default.device[1]), output_device)

    def get_device_samplerate(self):
        return int(sd.query_devices(device=sd.default.device[0])["default_samplerate"])

    def get_device_channels(self):
        max_input_channels = sd.query_devices(device=sd.default.device[0])[
            "max_input_channels"
        ]
        max_output_channels = sd.query_devices(device=sd.default.device[1])[
            "max_output_channels"
        ]
        return min(max_input_channels, max_output_channels, 2)

    # ---- lifecycle ----

    def start(self, data: dict) -> dict:
        """Validate config, load the model, start the stream.

        Returns {"samplerate": int, "delay_ms": int}.
        Raises ValueError for invalid configuration.
        """
        cfg = EngineConfig.from_dict(data)
        if not str(cfg.pth_path).strip():
            raise ValueError("Please select a .pth model file")
        if not str(cfg.index_path).strip():
            raise ValueError("Please select an .index file")
        if not os.path.exists(cfg.pth_path):
            raise ValueError("Model file does not exist: %s" % cfg.pth_path)
        if cfg.sg_input_device not in self.input_devices:
            raise ValueError("Unknown input device: %r" % cfg.sg_input_device)
        if cfg.sg_output_device not in self.output_devices:
            raise ValueError("Unknown output device: %r" % cfg.sg_output_device)
        self.stop()
        self.cfg = cfg
        self.function = data.get("function", "vc")
        self.set_devices(cfg.sg_input_device, cfg.sg_output_device)
        self.start_vc()
        self.delay_time = (
            self.stream.latency[-1] + cfg.block_time + cfg.crossfade_time + 0.01
        )
        if cfg.I_noise_reduce:
            self.delay_time += min(cfg.crossfade_time, 0.04)
        return {
            "samplerate": self.cfg.samplerate,
            "delay_ms": int(np.round(self.delay_time * 1000)),
        }

    def start_vc(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.rvc = rvc_for_realtime.RVC(
            self.cfg.pitch,
            self.cfg.formant,
            self.cfg.pth_path,
            self.cfg.index_path,
            self.cfg.index_rate,
            self.cfg.n_cpu,
            self.inp_q,
            self.opt_q,
            self.config,
            self.rvc if self.rvc is not None else None,
        )
        self.cfg.samplerate = (
            self.rvc.tgt_sr
            if self.cfg.sr_type == "sr_model"
            else self.get_device_samplerate()
        )
        self.cfg.channels = self.get_device_channels()
        self.zc = self.cfg.samplerate // 100
        self.block_frame = (
            int(np.round(self.cfg.block_time * self.cfg.samplerate / self.zc)) * self.zc
        )
        self.block_frame_16k = 160 * self.block_frame // self.zc
        self.crossfade_frame = (
            int(np.round(self.cfg.crossfade_time * self.cfg.samplerate / self.zc))
            * self.zc
        )
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = (
            int(np.round(self.cfg.extra_time * self.cfg.samplerate / self.zc)) * self.zc
        )
        self.input_wav: torch.Tensor = torch.zeros(
            self.extra_frame
            + self.crossfade_frame
            + self.sola_search_frame
            + self.block_frame,
            device=self.config.device,
            dtype=torch.float32,
        )
        self.input_wav_denoise: torch.Tensor = self.input_wav.clone()
        self.input_wav_res: torch.Tensor = torch.zeros(
            160 * self.input_wav.shape[0] // self.zc,
            device=self.config.device,
            dtype=torch.float32,
        )
        self.rms_buffer: np.ndarray = np.zeros(4 * self.zc, dtype="float32")
        self.sola_buffer: torch.Tensor = torch.zeros(
            self.sola_buffer_frame, device=self.config.device, dtype=torch.float32
        )
        self.nr_buffer: torch.Tensor = self.sola_buffer.clone()
        self.output_buffer: torch.Tensor = self.input_wav.clone()
        self.skip_head = self.extra_frame // self.zc
        self.return_length = (
            self.block_frame + self.sola_buffer_frame + self.sola_search_frame
        ) // self.zc
        self.fade_in_window: torch.Tensor = (
            torch.sin(
                0.5
                * np.pi
                * torch.linspace(
                    0.0,
                    1.0,
                    steps=self.sola_buffer_frame,
                    device=self.config.device,
                    dtype=torch.float32,
                )
            )
            ** 2
        )
        self.fade_out_window: torch.Tensor = 1 - self.fade_in_window
        self.resampler = tat.Resample(
            orig_freq=self.cfg.samplerate,
            new_freq=16000,
            dtype=torch.float32,
        ).to(self.config.device)
        if self.rvc.tgt_sr != self.cfg.samplerate:
            self.resampler2 = tat.Resample(
                orig_freq=self.rvc.tgt_sr,
                new_freq=self.cfg.samplerate,
                dtype=torch.float32,
            ).to(self.config.device)
        else:
            self.resampler2 = None
        self.tg = TorchGate(
            sr=self.cfg.samplerate, n_fft=4 * self.zc, prop_decrease=0.9
        ).to(self.config.device)
        self.start_stream()

    def start_stream(self):
        if not self.flag_vc:
            self.flag_vc = True
            if "WASAPI" in self.cfg.sg_hostapi and self.cfg.sg_wasapi_exclusive:
                extra_settings = sd.WasapiSettings(exclusive=True)
            else:
                extra_settings = None
            self.stream = sd.Stream(
                callback=self.audio_callback,
                blocksize=self.block_frame,
                samplerate=self.cfg.samplerate,
                channels=self.cfg.channels,
                dtype="float32",
                extra_settings=extra_settings,
            )
            self.stream.start()

    def stop(self):
        if self.flag_vc:
            self.flag_vc = False
            if self.stream is not None:
                self.stream.abort()
                self.stream.close()
                self.stream = None

    # ---- hot parameter updates (mirrors gui_v1's event_handler) ----

    def set_param(self, key, value) -> dict:
        """Update a parameter on a (possibly) running stream.

        Returns a dict of UI updates ({} or {"delay_ms": int}).
        Raises ValueError for keys that need a stream restart.
        """
        updates = {}
        if key == "threhold":
            self.cfg.threhold = value
        elif key == "pitch":
            self.cfg.pitch = value
            if self.rvc is not None:
                self.rvc.change_key(value)
        elif key == "formant":
            self.cfg.formant = value
            if self.rvc is not None:
                self.rvc.change_formant(value)
        elif key == "index_rate":
            self.cfg.index_rate = value
            if self.rvc is not None:
                self.rvc.change_index_rate(value)
        elif key == "rms_mix_rate":
            self.cfg.rms_mix_rate = value
        elif key == "f0method":
            self.cfg.f0method = value
        elif key == "I_noise_reduce":
            self.cfg.I_noise_reduce = bool(value)
            if self.stream is not None:
                self.delay_time += (1 if value else -1) * min(
                    self.cfg.crossfade_time, 0.04
                )
                updates["delay_ms"] = int(np.round(self.delay_time * 1000))
        elif key == "O_noise_reduce":
            self.cfg.O_noise_reduce = bool(value)
        elif key == "use_pv":
            self.cfg.use_pv = bool(value)
        elif key == "function":
            self.function = value
        else:
            raise ValueError("Parameter %r does not support hot update" % key)
        return updates

    # ---- audio processing (verbatim from gui_v1.GUI.audio_callback) ----

    def audio_callback(self, indata, outdata, frames, times, status):
        start_time = time.perf_counter()
        indata = librosa.to_mono(indata.T)
        if self.cfg.threhold > -60:
            indata = np.append(self.rms_buffer, indata)
            rms = librosa.feature.rms(
                y=indata, frame_length=4 * self.zc, hop_length=self.zc
            )[:, 2:]
            self.rms_buffer[:] = indata[-4 * self.zc :]
            indata = indata[2 * self.zc - self.zc // 2 :]
            db_threhold = librosa.amplitude_to_db(rms, ref=1.0)[0] < self.cfg.threhold
            for i in range(db_threhold.shape[0]):
                if db_threhold[i]:
                    indata[i * self.zc : (i + 1) * self.zc] = 0
            indata = indata[self.zc // 2 :]
        self.input_wav[: -self.block_frame] = self.input_wav[self.block_frame :].clone()
        self.input_wav[-indata.shape[0] :] = torch.from_numpy(indata).to(
            self.config.device
        )
        self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[
            self.block_frame_16k :
        ].clone()
        # input noise reduction and resampling
        if self.cfg.I_noise_reduce:
            self.input_wav_denoise[: -self.block_frame] = self.input_wav_denoise[
                self.block_frame :
            ].clone()
            input_wav = self.input_wav[-self.sola_buffer_frame - self.block_frame :]
            input_wav = self.tg(
                input_wav.unsqueeze(0), self.input_wav.unsqueeze(0)
            ).squeeze(0)
            input_wav[: self.sola_buffer_frame] *= self.fade_in_window
            input_wav[: self.sola_buffer_frame] += self.nr_buffer * self.fade_out_window
            self.input_wav_denoise[-self.block_frame :] = input_wav[: self.block_frame]
            self.nr_buffer[:] = input_wav[self.block_frame :]
            self.input_wav_res[-self.block_frame_16k - 160 :] = self.resampler(
                self.input_wav_denoise[-self.block_frame - 2 * self.zc :]
            )[160:]
        else:
            self.input_wav_res[-160 * (indata.shape[0] // self.zc + 1) :] = (
                self.resampler(self.input_wav[-indata.shape[0] - 2 * self.zc :])[160:]
            )
        # infer
        if self.function == "vc":
            infer_wav = self.rvc.infer(
                self.input_wav_res,
                self.block_frame_16k,
                self.skip_head,
                self.return_length,
                self.cfg.f0method,
            )
            if self.resampler2 is not None:
                infer_wav = self.resampler2(infer_wav)
        elif self.cfg.I_noise_reduce:
            infer_wav = self.input_wav_denoise[self.extra_frame :].clone()
        else:
            infer_wav = self.input_wav[self.extra_frame :].clone()
        # output noise reduction
        if self.cfg.O_noise_reduce and self.function == "vc":
            self.output_buffer[: -self.block_frame] = self.output_buffer[
                self.block_frame :
            ].clone()
            self.output_buffer[-self.block_frame :] = infer_wav[-self.block_frame :]
            infer_wav = self.tg(
                infer_wav.unsqueeze(0), self.output_buffer.unsqueeze(0)
            ).squeeze(0)
        # volume envelop mixing
        if self.cfg.rms_mix_rate < 1 and self.function == "vc":
            if self.cfg.I_noise_reduce:
                input_wav = self.input_wav_denoise[self.extra_frame :]
            else:
                input_wav = self.input_wav[self.extra_frame :]
            rms1 = librosa.feature.rms(
                y=input_wav[: infer_wav.shape[0]].cpu().numpy(),
                frame_length=4 * self.zc,
                hop_length=self.zc,
            )
            rms1 = torch.from_numpy(rms1).to(self.config.device)
            rms1 = F.interpolate(
                rms1.unsqueeze(0),
                size=infer_wav.shape[0] + 1,
                mode="linear",
                align_corners=True,
            )[0, 0, :-1]
            rms2 = librosa.feature.rms(
                y=infer_wav[:].cpu().numpy(),
                frame_length=4 * self.zc,
                hop_length=self.zc,
            )
            rms2 = torch.from_numpy(rms2).to(self.config.device)
            rms2 = F.interpolate(
                rms2.unsqueeze(0),
                size=infer_wav.shape[0] + 1,
                mode="linear",
                align_corners=True,
            )[0, 0, :-1]
            rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-3)
            infer_wav *= torch.pow(rms1 / rms2, torch.tensor(1 - self.cfg.rms_mix_rate))
        # SOLA algorithm from https://github.com/yxlllc/DDSP-SVC
        conv_input = infer_wav[
            None, None, : self.sola_buffer_frame + self.sola_search_frame
        ]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(
                conv_input**2,
                torch.ones(1, 1, self.sola_buffer_frame, device=self.config.device),
            )
            + 1e-8
        )
        if sys.platform == "darwin":
            _, sola_offset = torch.max(cor_nom[0, 0] / cor_den[0, 0])
            sola_offset = sola_offset.item()
        else:
            sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
        infer_wav = infer_wav[sola_offset:]
        if "privateuseone" in str(self.config.device) or not self.cfg.use_pv:
            infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
            infer_wav[: self.sola_buffer_frame] += (
                self.sola_buffer * self.fade_out_window
            )
        else:
            infer_wav[: self.sola_buffer_frame] = phase_vocoder(
                self.sola_buffer,
                infer_wav[: self.sola_buffer_frame],
                self.fade_out_window,
                self.fade_in_window,
            )
        self.sola_buffer[:] = infer_wav[
            self.block_frame : self.block_frame + self.sola_buffer_frame
        ]
        outdata[:] = (
            infer_wav[: self.block_frame].repeat(self.cfg.channels, 1).t().cpu().numpy()
        )
        total_time = time.perf_counter() - start_time
        if self.flag_vc and self.status_callback is not None:
            self.status_callback(
                {"type": "stats", "infer_time_ms": int(total_time * 1000)}
            )
