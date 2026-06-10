# Electron Realtime VC UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken `run-realtime.sh` flow with an Electron control panel driving a UI-agnostic Python realtime voice-conversion backend over a localhost WebSocket.

**Architecture:** The audio engine (sounddevice stream + SOLA pipeline + `infer.lib.rtrvc.RVC`) is extracted from `gui_v1.py` into `tools/realtime/engine.py` with zero UI imports. `tools/realtime/server.py` wraps it in a FastAPI WebSocket server on `127.0.0.1:6242`. An Electron app in `electron/` spawns the server and renders a full-parity control panel in vanilla HTML/JS. Audio never crosses the process boundary.

**Tech Stack:** Python 3.9 venv (torch, sounddevice, fastapi, uvicorn — all but sounddevice already installed), Electron ~33 (vanilla renderer, no bundler), WebSocket JSON protocol.

**Spec:** `docs/superpowers/specs/2026-06-10-electron-realtime-ui-design.md`

**Conventions for all tasks:**
- Run all commands from the repo root: `/Users/yuhaoli/Documents/rvc-polish`
- Python is always `.venv/bin/python`
- Commit messages: short lowercase summary line, then the trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

**Reference — message protocol (used by Tasks 3, 4, 6):**

Client → server: `{"type": "get_init"}`, `{"type": "update_devices", "hostapi": str}`, `{"type": "start", "config": {…}}`, `{"type": "stop"}`, `{"type": "set_param", "key": str, "value": any}`.
Server → client: `{"type": "init", "config": {…}, "hostapis": [], "input_devices": [], "output_devices": [], "n_cpu_max": int}`, `{"type": "devices", …same lists…}`, `{"type": "started", "samplerate": int, "delay_ms": int}`, `{"type": "stopped"}`, `{"type": "stats", "infer_time_ms": int}`, `{"type": "param_updated", "delay_ms": int}`, `{"type": "error", "message": str}`.

Config dict keys (same as gui_v1's `configs/inuse/config.json`, plus `function`): `pth_path, index_path, sg_hostapi, sg_wasapi_exclusive, sg_input_device, sg_output_device, sr_type, threhold, pitch, formant, index_rate, rms_mix_rate, block_time, crossfade_length, extra_time, n_cpu, f0method, use_jit, use_pv, function`. Note the historical spellings `threhold` and `crossfade_length` are kept for compatibility with gui_v1's config file.

---

### Task 1: Package skeleton + sounddevice dependency

**Files:**
- Create: `tools/realtime/__init__.py`
- Create: `tools/realtime/harvest_worker.py`

- [ ] **Step 1: Install sounddevice into the venv**

```bash
.venv/bin/python -m pip install sounddevice
```

Expected: `Successfully installed sounddevice-...` (it is absent today — it was only listed in the Windows realtime requirements files).

- [ ] **Step 2: Verify sounddevice can enumerate devices**

```bash
.venv/bin/python -c "import sounddevice as sd; print(len(sd.query_devices()), 'devices')"
```

Expected: a positive device count, no traceback.

- [ ] **Step 3: Create the package init**

Create `tools/realtime/__init__.py` with exactly:

```python
```

(empty file)

- [ ] **Step 4: Create the Harvest worker module**

This is gui_v1.py's `Harvest` class moved verbatim into an import-light module. It must stay light: multiprocessing's spawn start method (the macOS default) re-imports the modules needed to unpickle the process object in every worker, so this file must not import torch.

Create `tools/realtime/harvest_worker.py`:

```python
"""Harvest F0 worker process, extracted from gui_v1.py.

Kept import-light on purpose: the spawn start method re-imports this
module in every worker process, so heavy imports (torch, librosa) live
in run() or elsewhere.
"""

import multiprocessing


class Harvest(multiprocessing.Process):
    def __init__(self, inp_q, opt_q):
        multiprocessing.Process.__init__(self)
        self.inp_q = inp_q
        self.opt_q = opt_q

    def run(self):
        import numpy as np
        import pyworld

        while 1:
            idx, x, res_f0, n_cpu, ts = self.inp_q.get()
            f0, t = pyworld.harvest(
                x.astype(np.double),
                fs=16000,
                f0_ceil=1100,
                f0_floor=50,
                frame_period=10,
            )
            res_f0[idx] = f0
            if len(res_f0.keys()) >= n_cpu:
                self.opt_q.put(ts)
```

- [ ] **Step 5: Verify the module imports**

```bash
.venv/bin/python -c "from tools.realtime.harvest_worker import Harvest; print('OK')"
```

Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add tools/realtime/__init__.py tools/realtime/harvest_worker.py
git commit -m "add realtime package skeleton with harvest worker

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Realtime VC engine (UI-agnostic extraction of gui_v1.py)

**Files:**
- Create: `tools/realtime/engine.py`

This is a faithful extraction of `gui_v1.py`'s `GUIConfig` and `GUI` classes (lines 114–1068) with these mechanical changes and nothing else:
- `self.gui_config` → `self.cfg`; class renamed `GUIConfig` → `EngineConfig`
- `global flag_vc` → instance attribute `self.flag_vc`
- `self.window[...].update(...)` → `self.status_callback({...})` (a callable injected by the server; it is invoked from the PortAudio callback thread, so it must be thread-safe — the server handles that)
- `sg.popup(...)` validation → `raise ValueError(...)`
- The per-block `printt("sola_offset = %d", ...)` and `printt("Infer time: %.2f", ...)` lines are dropped (stats go through the callback instead; printing every 0.25s block would flood Electron's log capture)
- Harvest queues are constructor parameters instead of module globals

- [ ] **Step 1: Write engine.py**

Create `tools/realtime/engine.py`:

```python
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
    "infer_time_ms": 42}) from the PortAudio callback thread.
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
        return int(
            sd.query_devices(device=sd.default.device[0])["default_samplerate"]
        )

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
            int(np.round(self.cfg.block_time * self.cfg.samplerate / self.zc))
            * self.zc
        )
        self.block_frame_16k = 160 * self.block_frame // self.zc
        self.crossfade_frame = (
            int(np.round(self.cfg.crossfade_time * self.cfg.samplerate / self.zc))
            * self.zc
        )
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = (
            int(np.round(self.cfg.extra_time * self.cfg.samplerate / self.zc))
            * self.zc
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
            if (
                "WASAPI" in self.cfg.sg_hostapi
                and self.cfg.sg_wasapi_exclusive
            ):
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
            db_threhold = (
                librosa.amplitude_to_db(rms, ref=1.0)[0] < self.cfg.threhold
            )
            for i in range(db_threhold.shape[0]):
                if db_threhold[i]:
                    indata[i * self.zc : (i + 1) * self.zc] = 0
            indata = indata[self.zc // 2 :]
        self.input_wav[: -self.block_frame] = self.input_wav[
            self.block_frame :
        ].clone()
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
            input_wav[: self.sola_buffer_frame] += (
                self.nr_buffer * self.fade_out_window
            )
            self.input_wav_denoise[-self.block_frame :] = input_wav[
                : self.block_frame
            ]
            self.nr_buffer[:] = input_wav[self.block_frame :]
            self.input_wav_res[-self.block_frame_16k - 160 :] = self.resampler(
                self.input_wav_denoise[-self.block_frame - 2 * self.zc :]
            )[160:]
        else:
            self.input_wav_res[-160 * (indata.shape[0] // self.zc + 1) :] = (
                self.resampler(self.input_wav[-indata.shape[0] - 2 * self.zc :])[
                    160:
                ]
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
            infer_wav *= torch.pow(
                rms1 / rms2, torch.tensor(1 - self.cfg.rms_mix_rate)
            )
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
            infer_wav[: self.block_frame]
            .repeat(self.cfg.channels, 1)
            .t()
            .cpu()
            .numpy()
        )
        total_time = time.perf_counter() - start_time
        if self.flag_vc and self.status_callback is not None:
            self.status_callback(
                {"type": "stats", "infer_time_ms": int(total_time * 1000)}
            )
```

- [ ] **Step 2: Smoke-test import and device enumeration**

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -c "
from multiprocessing import Queue
from tools.realtime.engine import RealtimeVC, EngineConfig
e = RealtimeVC(Queue(), Queue())
assert isinstance(e.hostapis, list) and len(e.hostapis) > 0, e.hostapis
assert isinstance(e.input_devices, list)
assert isinstance(e.output_devices, list)
cfg = EngineConfig.from_dict({'crossfade_length': 0.07, 'pitch': 3})
assert cfg.crossfade_time == 0.07 and cfg.pitch == 3
print('hostapis:', e.hostapis)
print('inputs:', len(e.input_devices), 'outputs:', len(e.output_devices))
print('ENGINE OK')
"
```

Expected: `ENGINE OK` (torch import takes ~10-30s; Config() prints device-detection lines — that is normal).

- [ ] **Step 3: Commit**

```bash
git add tools/realtime/engine.py
git commit -m "extract ui-agnostic realtime vc engine from gui_v1

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Protocol smoke test (written first, fails until Task 4)

**Files:**
- Create: `tools/realtime/test_protocol.py`

Plain Python (no pytest — it is not installed in the venv). Uses the already-installed `websockets` package.

- [ ] **Step 1: Write the test**

Create `tools/realtime/test_protocol.py`:

```python
"""Headless smoke test for the realtime control server protocol.

Run: .venv/bin/python tools/realtime/test_protocol.py

Starts the server as a subprocess, then checks:
  1. get_init returns device lists and the saved config
  2. start with a bogus model path returns a clean error (no crash)
  3. stop returns stopped
"""

import asyncio
import json
import os
import subprocess
import sys
import threading

NOW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SERVER = os.path.join(NOW_DIR, "tools", "realtime", "server.py")


def start_server():
    proc = subprocess.Popen(
        [sys.executable, SERVER, "--port", "6342"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=NOW_DIR,
    )
    port = None
    for line in proc.stdout:
        sys.stdout.write("[server] " + line)
        if line.startswith("REALTIME_SERVER_PORT="):
            port = int(line.strip().split("=", 1)[1])
            break
        if proc.poll() is not None:
            break
    if port is None:
        raise RuntimeError("Server exited before announcing its port")
    # keep draining stdout so the pipe never fills up
    threading.Thread(target=proc.stdout.read, daemon=True).start()
    return proc, port


async def connect_with_retry(uri):
    import websockets

    for _ in range(40):
        try:
            return await websockets.connect(uri)
        except OSError:
            await asyncio.sleep(0.5)
    raise RuntimeError("Could not connect to %s" % uri)


async def run_checks(port):
    ws = await connect_with_retry("ws://127.0.0.1:%d/ws" % port)
    try:
        await ws.send(json.dumps({"type": "get_init"}))
        init = json.loads(await ws.recv())
        assert init["type"] == "init", init
        assert isinstance(init["hostapis"], list) and init["hostapis"], init
        assert isinstance(init["input_devices"], list), init
        assert isinstance(init["output_devices"], list), init
        assert "pth_path" in init["config"], init
        print(
            "get_init OK: %d input / %d output devices"
            % (len(init["input_devices"]), len(init["output_devices"]))
        )

        bogus = dict(init["config"])
        bogus["pth_path"] = "/nonexistent/model.pth"
        bogus["index_path"] = "/nonexistent/model.index"
        await ws.send(json.dumps({"type": "start", "config": bogus}))
        resp = json.loads(await ws.recv())
        assert resp["type"] == "error", resp
        print("start with bogus path -> clean error: %s" % resp["message"])

        await ws.send(json.dumps({"type": "stop"}))
        resp = json.loads(await ws.recv())
        assert resp["type"] == "stopped", resp
        print("stop OK")
    finally:
        await ws.close()


def main():
    proc, port = start_server()
    try:
        asyncio.run(run_checks(port))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("PROTOCOL SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it to verify it fails (server does not exist yet)**

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python tools/realtime/test_protocol.py
```

Expected: FAIL — `RuntimeError: Server exited before announcing its port` (python printed `can't open file .../server.py`).

- [ ] **Step 3: Commit the test**

```bash
git add tools/realtime/test_protocol.py
git commit -m "add protocol smoke test for realtime server

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: WebSocket control server

**Files:**
- Create: `tools/realtime/server.py`

Design notes baked into the code below:
- Top level stays import-light (multiprocessing spawn re-imports the main module in every Harvest worker; heavy imports live inside `main()`).
- `configs.config.Config` runs its own argparse over `sys.argv`, so `main()` parses `--host/--port` first and then truncates `sys.argv` before importing the engine.
- The chosen port is announced as `REALTIME_SERVER_PORT=<port>` on stdout (flushed) so both Electron and the smoke test can parse it; if the requested port is busy the next 9 are tried.
- `push_status` is handed to the engine and called from the PortAudio thread; it hops onto the asyncio loop with `call_soon_threadsafe`.

- [ ] **Step 1: Write server.py**

Create `tools/realtime/server.py`:

```python
"""WebSocket control server for the realtime VC engine.

Run: .venv/bin/python tools/realtime/server.py [--port 6242]

Announces the chosen port on stdout as REALTIME_SERVER_PORT=<port>.
Heavy imports happen inside main(): the multiprocessing spawn start
method re-imports this module in every Harvest worker, and those
workers must not pay for torch/fastapi imports.
"""

import argparse
import multiprocessing
import os
import sys

os.environ["OMP_NUM_THREADS"] = "4"
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

NOW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.path.join(NOW_DIR, "configs", "inuse", "config.json")

DEFAULT_CONFIG = {
    "pth_path": "",
    "index_path": "",
    "sg_hostapi": "",
    "sg_wasapi_exclusive": False,
    "sg_input_device": "",
    "sg_output_device": "",
    "sr_type": "sr_model",
    "threhold": -60,
    "pitch": 0,
    "formant": 0.0,
    "index_rate": 0,
    "rms_mix_rate": 0,
    "block_time": 0.25,
    "crossfade_length": 0.05,
    "extra_time": 2.5,
    "n_cpu": 4,
    "f0method": "rmvpe",
    "use_jit": False,
    "use_pv": False,
    "function": "vc",
}


def pick_port(host, start_port):
    import socket

    for port in range(start_port, start_port + 10):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(
        "No free port in range %d-%d" % (start_port, start_port + 9)
    )


def load_saved_config():
    import json

    try:
        with open(CONFIG_PATH, "r") as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {}
    merged = dict(DEFAULT_CONFIG)
    merged.update({k: v for k, v in data.items() if k in DEFAULT_CONFIG})
    return merged


def save_config(data):
    import json

    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    keep = {k: data.get(k, DEFAULT_CONFIG[k]) for k in DEFAULT_CONFIG}
    with open(CONFIG_PATH, "w") as f:
        json.dump(keep, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6242)
    args = parser.parse_args()
    # configs.config.Config runs its own argparse; hide our args from it
    sys.argv = sys.argv[:1]

    os.chdir(NOW_DIR)
    if NOW_DIR not in sys.path:
        sys.path.append(NOW_DIR)

    n_cpu = min(multiprocessing.cpu_count(), 8)
    inp_q = multiprocessing.Queue()
    opt_q = multiprocessing.Queue()
    from tools.realtime.harvest_worker import Harvest

    for _ in range(n_cpu):
        p = Harvest(inp_q, opt_q)
        p.daemon = True
        p.start()

    import asyncio
    import json
    import traceback

    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect

    from tools.realtime.engine import RealtimeVC

    app = FastAPI()
    clients = set()
    state = {"loop": None}

    def push_status(msg):
        loop = state["loop"]
        if loop is not None:
            loop.call_soon_threadsafe(_broadcast, msg)

    def _broadcast(msg):
        text = json.dumps(msg)
        for ws in list(clients):
            asyncio.ensure_future(_send_safe(ws, text))

    async def _send_safe(ws, text):
        try:
            await ws.send_text(text)
        except Exception:
            clients.discard(ws)

    engine = RealtimeVC(inp_q, opt_q, status_callback=push_status)

    def devices_payload():
        return {
            "hostapis": engine.hostapis,
            "input_devices": engine.input_devices,
            "output_devices": engine.output_devices,
        }

    async def handle(msg):
        mtype = msg.get("type")
        if mtype == "get_init":
            saved = load_saved_config()
            if saved["sg_hostapi"] in engine.hostapis:
                engine.update_devices(hostapi_name=saved["sg_hostapi"])
            payload = {"type": "init", "config": saved, "n_cpu_max": n_cpu}
            payload.update(devices_payload())
            return payload
        if mtype == "update_devices":
            engine.update_devices(hostapi_name=msg.get("hostapi"))
            payload = {"type": "devices"}
            payload.update(devices_payload())
            return payload
        if mtype == "start":
            data = msg.get("config", {})
            try:
                result = await asyncio.to_thread(engine.start, data)
            except Exception as exc:
                traceback.print_exc()
                return {"type": "error", "message": str(exc)}
            save_config(data)
            return {"type": "started", **result}
        if mtype == "stop":
            await asyncio.to_thread(engine.stop)
            return {"type": "stopped"}
        if mtype == "set_param":
            try:
                updates = engine.set_param(msg.get("key"), msg.get("value"))
            except Exception as exc:
                return {"type": "error", "message": str(exc)}
            if updates:
                return {"type": "param_updated", **updates}
            return None
        return {"type": "error", "message": "Unknown message type: %r" % mtype}

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        state["loop"] = asyncio.get_running_loop()
        clients.add(ws)
        try:
            while True:
                msg = json.loads(await ws.receive_text())
                resp = await handle(msg)
                if resp is not None:
                    await ws.send_text(json.dumps(resp))
        except WebSocketDisconnect:
            pass
        finally:
            clients.discard(ws)

    port = pick_port(args.host, args.port)
    print("REALTIME_SERVER_PORT=%d" % port, flush=True)
    uvicorn.run(app, host=args.host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
```

(Note: `asyncio.to_thread` requires Python ≥3.9; the venv is 3.9, and `run-realtime.sh` already enforces that.)

- [ ] **Step 2: Run the smoke test to verify it passes**

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python tools/realtime/test_protocol.py
```

Expected output ends with:

```
get_init OK: ... input / ... output devices
start with bogus path -> clean error: Model file does not exist: /nonexistent/model.pth
stop OK
PROTOCOL SMOKE TEST PASSED
```

(Server startup takes ~10-40s while torch imports; the test waits for the port line.)

- [ ] **Step 3: Commit**

```bash
git add tools/realtime/server.py
git commit -m "add websocket control server for realtime vc

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Electron scaffold (main process, preload, package.json)

**Files:**
- Create: `electron/package.json`
- Create: `electron/main.js`
- Create: `electron/preload.js`
- Modify: `.gitignore` (append one line)

- [ ] **Step 1: Create package.json**

Create `electron/package.json`:

```json
{
  "name": "rvc-realtime-ui",
  "version": "1.0.0",
  "private": true,
  "description": "Electron control panel for RVC realtime voice conversion",
  "main": "main.js",
  "scripts": {
    "start": "electron ."
  },
  "devDependencies": {
    "electron": "^33.0.0"
  }
}
```

- [ ] **Step 2: Create preload.js**

Create `electron/preload.js`:

```js
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("rvcApi", {
  getPort: () => ipcRenderer.invoke("get-port"),
  pickFile: (opts) => ipcRenderer.invoke("pick-file", opts),
});
```

- [ ] **Step 3: Create main.js**

The window opens immediately with a "Starting backend…" status (torch takes ~10-40s to import); the renderer polls `get-port` until the backend has announced its port.

Create `electron/main.js`:

```js
const {
  app,
  BrowserWindow,
  dialog,
  ipcMain,
  systemPreferences,
} = require("electron");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");

const repoRoot = path.resolve(__dirname, "..");
const pythonBin =
  process.env.RVC_PYTHON || path.join(repoRoot, ".venv", "bin", "python");
const serverScript = path.join(repoRoot, "tools", "realtime", "server.py");

let win = null;
let backend = null;
let backendPort = null;
let quitting = false;
const stderrTail = [];

function rememberStderr(chunk) {
  for (const line of chunk.toString().split("\n")) {
    if (line.trim()) stderrTail.push(line);
  }
  while (stderrTail.length > 30) stderrTail.shift();
}

function startBackend() {
  return new Promise((resolve, reject) => {
    if (!fs.existsSync(pythonBin)) {
      reject(
        new Error(`Python not found at ${pythonBin}. Run ./run-realtime.sh first.`)
      );
      return;
    }
    backend = spawn(pythonBin, [serverScript], { cwd: repoRoot });
    const timeout = setTimeout(() => {
      reject(
        new Error(
          "Backend did not start within 180s.\n" + stderrTail.join("\n")
        )
      );
    }, 180000);
    let buffer = "";
    backend.stdout.on("data", (chunk) => {
      buffer += chunk.toString();
      let idx;
      while ((idx = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 1);
        if (line) console.log("[backend]", line);
        const m = line.match(/^REALTIME_SERVER_PORT=(\d+)$/);
        if (m) {
          clearTimeout(timeout);
          backendPort = parseInt(m[1], 10);
          resolve(backendPort);
        }
      }
    });
    backend.stderr.on("data", (chunk) => {
      rememberStderr(chunk);
      console.error("[backend]", chunk.toString().trimEnd());
    });
    backend.on("exit", (code) => {
      backend = null;
      if (!quitting) {
        dialog.showErrorBox(
          "RVC backend stopped",
          `The Python backend exited (code ${code}).\n\nLast output:\n` +
            stderrTail.join("\n")
        );
        app.quit();
      }
    });
  });
}

function createWindow() {
  win = new BrowserWindow({
    width: 1100,
    height: 880,
    title: "RVC Realtime Voice Conversion",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
    },
  });
  win.loadFile(path.join(__dirname, "renderer", "index.html"));
}

app.whenReady().then(async () => {
  if (process.platform === "darwin") {
    await systemPreferences.askForMediaAccess("microphone");
  }
  ipcMain.handle("get-port", () => backendPort);
  ipcMain.handle("pick-file", async (event, opts) => {
    const result = await dialog.showOpenDialog(win, {
      properties: ["openFile"],
      defaultPath: opts.defaultDir
        ? path.join(repoRoot, opts.defaultDir)
        : repoRoot,
      filters: opts.filters || [],
    });
    return result.canceled ? null : result.filePaths[0];
  });
  createWindow();
  try {
    await startBackend();
  } catch (err) {
    dialog.showErrorBox(
      "RVC backend failed to start",
      String((err && err.message) || err)
    );
    app.quit();
  }
});

app.on("before-quit", () => {
  quitting = true;
  if (backend) backend.kill();
});

app.on("window-all-closed", () => {
  app.quit();
});
```

- [ ] **Step 4: Ignore node_modules**

Append to `.gitignore`:

```
electron/node_modules/
```

- [ ] **Step 5: Install Electron**

```bash
npm install --prefix electron
```

Expected: `added ... packages` (downloads the Electron binary; can take a few minutes).

- [ ] **Step 6: Verify Electron runs**

```bash
./electron/node_modules/.bin/electron --version
```

Expected: a version string like `v33.x.x`.

(Don't `npm start` yet — `renderer/index.html` doesn't exist until Task 6.)

- [ ] **Step 7: Commit**

```bash
git add electron/package.json electron/main.js electron/preload.js electron/package-lock.json .gitignore
git commit -m "add electron scaffold for realtime ui

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Renderer (full-parity control panel)

**Files:**
- Create: `electron/renderer/index.html`
- Create: `electron/renderer/style.css`
- Create: `electron/renderer/app.js`

Control IDs deliberately match the backend config keys (including the historical `threhold` / `crossfade_length` spellings) so `currentConfig()` can read them generically. Slider ranges are copied from gui_v1's layout: threhold −60..0 step 1, pitch −16..16 step 1, formant −2..2 step 0.05, index_rate 0..1 step 0.01, rms_mix_rate 0..1 step 0.01, block_time 0.02..1.5 step 0.01, n_cpu 1..8 step 1 (max set from `n_cpu_max` at init), crossfade_length 0.01..0.15 step 0.01, extra_time 0.05..5 step 0.01.

- [ ] **Step 1: Create index.html**

Create `electron/renderer/index.html`:

```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <meta http-equiv="Content-Security-Policy"
        content="default-src 'self'; connect-src ws://127.0.0.1:*; style-src 'self'" />
  <title>RVC Realtime Voice Conversion</title>
  <link rel="stylesheet" href="style.css" />
</head>
<body>
  <div id="banner" class="banner hidden">
    <span id="banner_text"></span>
    <button id="banner_close" title="Dismiss">&times;</button>
  </div>

  <main>
    <section class="card">
      <h2>Model</h2>
      <div class="row">
        <label>.pth file</label>
        <input id="pth_path" type="text" readonly placeholder="Select a voice model (.pth)" />
        <button id="browse_pth">Browse&hellip;</button>
      </div>
      <div class="row">
        <label>.index file</label>
        <input id="index_path" type="text" readonly placeholder="Select a feature index (.index)" />
        <button id="browse_index">Browse&hellip;</button>
      </div>
    </section>

    <section class="card">
      <h2>Audio devices</h2>
      <div class="row">
        <label>Host API</label>
        <select id="sg_hostapi"></select>
        <label class="check"><input id="sg_wasapi_exclusive" type="checkbox" /> Exclusive WASAPI mode</label>
      </div>
      <div class="row">
        <label>Input device</label>
        <select id="sg_input_device"></select>
      </div>
      <div class="row">
        <label>Output device</label>
        <select id="sg_output_device"></select>
      </div>
      <div class="row">
        <button id="reload_devices">Reload device list</button>
        <label class="check"><input type="radio" name="sr_type" id="sr_model" checked /> Use model sample rate</label>
        <label class="check"><input type="radio" name="sr_type" id="sr_device" /> Use device sample rate</label>
        <span class="stat">Sample rate: <span id="sr_stream">&ndash;</span></span>
      </div>
    </section>

    <div class="columns">
      <section class="card">
        <h2>Inference settings</h2>
        <div class="row slider-row">
          <label>Response threshold</label>
          <input type="range" id="threhold" min="-60" max="0" step="1" value="-60" />
          <span class="val" id="threhold_val">-60</span>
        </div>
        <div class="row slider-row">
          <label>Pitch</label>
          <input type="range" id="pitch" min="-16" max="16" step="1" value="0" />
          <span class="val" id="pitch_val">0</span>
        </div>
        <div class="row slider-row">
          <label>Formant (gender factor)</label>
          <input type="range" id="formant" min="-2" max="2" step="0.05" value="0" />
          <span class="val" id="formant_val">0.00</span>
        </div>
        <div class="row slider-row">
          <label>Index rate</label>
          <input type="range" id="index_rate" min="0" max="1" step="0.01" value="0" />
          <span class="val" id="index_rate_val">0.00</span>
        </div>
        <div class="row slider-row">
          <label>Loudness factor</label>
          <input type="range" id="rms_mix_rate" min="0" max="1" step="0.01" value="0" />
          <span class="val" id="rms_mix_rate_val">0.00</span>
        </div>
        <div class="row">
          <label>Pitch algorithm</label>
          <span class="radio-group">
            <label class="check"><input type="radio" name="f0method" value="pm" /> pm</label>
            <label class="check"><input type="radio" name="f0method" value="harvest" /> harvest</label>
            <label class="check"><input type="radio" name="f0method" value="crepe" /> crepe</label>
            <label class="check"><input type="radio" name="f0method" value="rmvpe" checked /> rmvpe</label>
            <label class="check"><input type="radio" name="f0method" value="fcpe" /> fcpe</label>
          </span>
        </div>
      </section>

      <section class="card">
        <h2>Performance</h2>
        <div class="row slider-row">
          <label>Block time (s)</label>
          <input type="range" id="block_time" min="0.02" max="1.5" step="0.01" value="0.25" />
          <span class="val" id="block_time_val">0.25</span>
        </div>
        <div class="row slider-row">
          <label>Harvest processes</label>
          <input type="range" id="n_cpu" min="1" max="8" step="1" value="4" />
          <span class="val" id="n_cpu_val">4</span>
        </div>
        <div class="row slider-row">
          <label>Crossfade length (s)</label>
          <input type="range" id="crossfade_length" min="0.01" max="0.15" step="0.01" value="0.05" />
          <span class="val" id="crossfade_length_val">0.05</span>
        </div>
        <div class="row slider-row">
          <label>Extra inference time (s)</label>
          <input type="range" id="extra_time" min="0.05" max="5" step="0.01" value="2.5" />
          <span class="val" id="extra_time_val">2.50</span>
        </div>
        <div class="row">
          <label class="check"><input id="I_noise_reduce" type="checkbox" /> Input noise reduction</label>
          <label class="check"><input id="O_noise_reduce" type="checkbox" /> Output noise reduction</label>
          <label class="check"><input id="use_pv" type="checkbox" /> Phase vocoder</label>
        </div>
      </section>
    </div>

    <section class="card control-bar">
      <button id="start_vc" class="primary">Start conversion</button>
      <button id="stop_vc" disabled>Stop</button>
      <span class="radio-group">
        <label class="check"><input type="radio" name="function" value="im" /> Monitor input</label>
        <label class="check"><input type="radio" name="function" value="vc" checked /> Convert output</label>
      </span>
      <span class="stat">Algorithm latency: <span id="delay_time">0</span> ms</span>
      <span class="stat">Inference time: <span id="infer_time">0</span> ms</span>
      <span class="stat" id="conn_status">Starting Python backend&hellip;</span>
    </section>
  </main>

  <script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create style.css**

Create `electron/renderer/style.css`:

```css
:root {
  --bg: #14161a;
  --card: #1d2026;
  --border: #2c313a;
  --text: #e6e8eb;
  --muted: #9aa3af;
  --accent: #4f8cff;
  --danger: #b33939;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 13px;
}

main {
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 12px 16px;
}

.card h2 {
  margin: 0 0 10px;
  font-size: 13px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--muted);
}

.columns {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}

.row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
  flex-wrap: wrap;
}

.row:last-child {
  margin-bottom: 0;
}

.row > label:first-child {
  flex: 0 0 170px;
  color: var(--muted);
}

.row input[type="text"] {
  flex: 1;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text);
  padding: 6px 8px;
  font-size: 12px;
}

.row select {
  flex: 1;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text);
  padding: 6px 8px;
}

.slider-row input[type="range"] {
  flex: 1;
  accent-color: var(--accent);
}

.slider-row .val {
  flex: 0 0 48px;
  text-align: right;
  font-variant-numeric: tabular-nums;
  color: var(--text);
}

button {
  background: #2a2f37;
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text);
  padding: 6px 14px;
  cursor: pointer;
}

button:hover:not(:disabled) {
  border-color: var(--accent);
}

button:disabled {
  opacity: 0.45;
  cursor: default;
}

button.primary {
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
  font-weight: 600;
}

.check {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  color: var(--text);
  white-space: nowrap;
}

.radio-group {
  display: inline-flex;
  gap: 14px;
  flex-wrap: wrap;
}

.control-bar {
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
}

.stat {
  color: var(--muted);
}

.stat span {
  color: var(--text);
  font-variant-numeric: tabular-nums;
}

.banner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  background: var(--danger);
  color: #fff;
  padding: 8px 14px;
  position: sticky;
  top: 0;
  z-index: 10;
}

.banner button {
  background: transparent;
  border: none;
  color: #fff;
  font-size: 16px;
  padding: 0 4px;
}

.hidden {
  display: none;
}
```

- [ ] **Step 3: Create app.js**

Create `electron/renderer/app.js`:

```js
const $ = (id) => document.getElementById(id);

const SLIDERS = [
  { id: "threhold", decimals: 0 },
  { id: "pitch", decimals: 0 },
  { id: "formant", decimals: 2 },
  { id: "index_rate", decimals: 2 },
  { id: "rms_mix_rate", decimals: 2 },
  { id: "block_time", decimals: 2 },
  { id: "n_cpu", decimals: 0 },
  { id: "crossfade_length", decimals: 2 },
  { id: "extra_time", decimals: 2 },
];

// Params the backend hot-updates on a running stream (mirrors gui_v1)
const HOT_PARAMS = new Set([
  "threhold",
  "pitch",
  "formant",
  "index_rate",
  "rms_mix_rate",
]);

// Controls that require a stream restart, locked while running
const COLD_CONTROLS = [
  "browse_pth",
  "browse_index",
  "sg_hostapi",
  "sg_wasapi_exclusive",
  "sg_input_device",
  "sg_output_device",
  "reload_devices",
  "sr_model",
  "sr_device",
  "block_time",
  "n_cpu",
  "crossfade_length",
  "extra_time",
];

let ws = null;
let running = false;

function setStatus(text) {
  $("conn_status").textContent = text;
}

function showError(text) {
  $("banner_text").textContent = text;
  $("banner").classList.remove("hidden");
}

function send(msg) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(msg));
  }
}

function sliderLabel(id, decimals) {
  $(id + "_val").textContent = Number($(id).value).toFixed(decimals);
}

function fillSelect(sel, options, selected) {
  sel.innerHTML = "";
  for (const name of options) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    if (name === selected) opt.selected = true;
    sel.appendChild(opt);
  }
}

function checkedValue(name) {
  const el = document.querySelector(`input[name="${name}"]:checked`);
  return el ? el.value : null;
}

function setRadio(name, value) {
  const el = document.querySelector(`input[name="${name}"][value="${value}"]`);
  if (el) el.checked = true;
}

function currentConfig() {
  return {
    pth_path: $("pth_path").value,
    index_path: $("index_path").value,
    sg_hostapi: $("sg_hostapi").value,
    sg_wasapi_exclusive: $("sg_wasapi_exclusive").checked,
    sg_input_device: $("sg_input_device").value,
    sg_output_device: $("sg_output_device").value,
    sr_type: $("sr_model").checked ? "sr_model" : "sr_device",
    threhold: parseFloat($("threhold").value),
    pitch: parseFloat($("pitch").value),
    formant: parseFloat($("formant").value),
    index_rate: parseFloat($("index_rate").value),
    rms_mix_rate: parseFloat($("rms_mix_rate").value),
    block_time: parseFloat($("block_time").value),
    crossfade_length: parseFloat($("crossfade_length").value),
    extra_time: parseFloat($("extra_time").value),
    n_cpu: parseFloat($("n_cpu").value),
    f0method: checkedValue("f0method"),
    I_noise_reduce: $("I_noise_reduce").checked,
    O_noise_reduce: $("O_noise_reduce").checked,
    use_pv: $("use_pv").checked,
    use_jit: false,
    function: checkedValue("function"),
  };
}

function applyConfig(cfg) {
  $("pth_path").value = cfg.pth_path || "";
  $("index_path").value = cfg.index_path || "";
  $("sg_wasapi_exclusive").checked = !!cfg.sg_wasapi_exclusive;
  $("sr_model").checked = cfg.sr_type !== "sr_device";
  $("sr_device").checked = cfg.sr_type === "sr_device";
  for (const { id, decimals } of SLIDERS) {
    if (cfg[id] !== undefined) {
      $(id).value = cfg[id];
      sliderLabel(id, decimals);
    }
  }
  if (cfg.f0method) setRadio("f0method", cfg.f0method);
  if (cfg.function) setRadio("function", cfg.function);
  $("I_noise_reduce").checked = !!cfg.I_noise_reduce;
  $("O_noise_reduce").checked = !!cfg.O_noise_reduce;
  $("use_pv").checked = !!cfg.use_pv;
}

function setRunning(value) {
  running = value;
  $("start_vc").disabled = value;
  $("stop_vc").disabled = !value;
  for (const id of COLD_CONTROLS) {
    $(id).disabled = value;
  }
}

function handleMessage(msg) {
  switch (msg.type) {
    case "init":
      fillSelect($("sg_hostapi"), msg.hostapis, msg.config.sg_hostapi);
      fillSelect($("sg_input_device"), msg.input_devices, msg.config.sg_input_device);
      fillSelect($("sg_output_device"), msg.output_devices, msg.config.sg_output_device);
      $("n_cpu").max = msg.n_cpu_max;
      applyConfig(msg.config);
      setStatus("Ready");
      break;
    case "devices":
      fillSelect($("sg_hostapi"), msg.hostapis, $("sg_hostapi").value);
      fillSelect($("sg_input_device"), msg.input_devices, null);
      fillSelect($("sg_output_device"), msg.output_devices, null);
      break;
    case "started":
      setRunning(true);
      $("sr_stream").textContent = msg.samplerate;
      $("delay_time").textContent = msg.delay_ms;
      setStatus("Converting");
      break;
    case "stopped":
      setRunning(false);
      setStatus("Ready");
      break;
    case "stats":
      $("infer_time").textContent = msg.infer_time_ms;
      break;
    case "param_updated":
      if (msg.delay_ms !== undefined) {
        $("delay_time").textContent = msg.delay_ms;
      }
      break;
    case "error":
      showError(msg.message);
      if (!running) setStatus("Ready");
      break;
  }
}

function openSocket(port) {
  ws = new WebSocket(`ws://127.0.0.1:${port}/ws`);
  let opened = false;
  ws.onopen = () => {
    opened = true;
    setStatus("Loading saved settings…");
    send({ type: "get_init" });
  };
  ws.onmessage = (e) => handleMessage(JSON.parse(e.data));
  ws.onclose = () => {
    if (!opened) {
      // uvicorn may not be accepting connections yet; retry
      setTimeout(() => openSocket(port), 1000);
    } else {
      setRunning(false);
      setStatus("Backend disconnected");
      showError("Lost connection to the Python backend.");
    }
  };
}

async function connect() {
  setStatus("Starting Python backend… (first start takes a while)");
  let port = await window.rvcApi.getPort();
  while (!port) {
    await new Promise((r) => setTimeout(r, 500));
    port = await window.rvcApi.getPort();
  }
  openSocket(port);
}

function wireEvents() {
  $("browse_pth").onclick = async () => {
    const file = await window.rvcApi.pickFile({
      defaultDir: "assets/weights",
      filters: [{ name: "RVC model", extensions: ["pth"] }],
    });
    if (file) $("pth_path").value = file;
  };
  $("browse_index").onclick = async () => {
    const file = await window.rvcApi.pickFile({
      defaultDir: "logs",
      filters: [{ name: "Feature index", extensions: ["index"] }],
    });
    if (file) $("index_path").value = file;
  };
  $("sg_hostapi").onchange = () =>
    send({ type: "update_devices", hostapi: $("sg_hostapi").value });
  $("reload_devices").onclick = () =>
    send({ type: "update_devices", hostapi: $("sg_hostapi").value });
  $("start_vc").onclick = () => {
    const cfg = currentConfig();
    if (!cfg.pth_path) {
      showError("Select a .pth model file first.");
      return;
    }
    if (!cfg.index_path) {
      showError("Select an .index file first.");
      return;
    }
    $("banner").classList.add("hidden");
    setStatus("Loading model…");
    send({ type: "start", config: cfg });
  };
  $("stop_vc").onclick = () => send({ type: "stop" });

  for (const { id, decimals } of SLIDERS) {
    $(id).oninput = () => {
      sliderLabel(id, decimals);
      if (running && HOT_PARAMS.has(id)) {
        send({ type: "set_param", key: id, value: parseFloat($(id).value) });
      }
    };
  }
  for (const radio of document.querySelectorAll('input[name="f0method"]')) {
    radio.onchange = () => {
      if (running) send({ type: "set_param", key: "f0method", value: radio.value });
    };
  }
  for (const radio of document.querySelectorAll('input[name="function"]')) {
    radio.onchange = () => {
      if (running) send({ type: "set_param", key: "function", value: radio.value });
    };
  }
  for (const id of ["I_noise_reduce", "O_noise_reduce", "use_pv"]) {
    $(id).onchange = () => {
      if (running) send({ type: "set_param", key: id, value: $(id).checked });
    };
  }
  $("banner_close").onclick = () => $("banner").classList.add("hidden");
}

wireEvents();
setRunning(false);
connect();
```

- [ ] **Step 4: Launch and verify end-to-end startup**

```bash
npm start --prefix electron
```

Expected within ~60s:
1. A window titled "RVC Realtime Voice Conversion" opens with all four sections and the control bar.
2. The status goes "Starting Python backend…" → "Ready"; the terminal shows `[backend] REALTIME_SERVER_PORT=6242`.
3. Host API / input / output dropdowns are populated with real device names.
4. No uncaught exceptions in the terminal.

Quit with Cmd+Q and confirm the python process exits too (`pgrep -f tools/realtime/server.py` prints nothing).

- [ ] **Step 5: Commit**

```bash
git add electron/renderer
git commit -m "add electron renderer with full gui_v1 parity

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Wire up run-realtime.sh and clean up the misleading stub

**Files:**
- Modify: `run-realtime.sh` (replace the final "Run the real-time voice conversion" section)
- Modify: `tools/rvc_for_realtime.py:457-461` (the `if __name__ == '__main__':` block)

- [ ] **Step 1: Rewrite run-realtime.sh**

Replace the entire contents of `run-realtime.sh` with:

```sh
#!/bin/sh

cd "$(dirname "$0")"

if [ "$(uname)" = "Darwin" ]; then
  # macOS specific env:
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
elif [ "$(uname)" != "Linux" ]; then
  echo "Unsupported operating system."
  exit 1
fi

find_python() {
  if [ -n "${PYTHON:-}" ] && command -v "${PYTHON}" >/dev/null 2>&1; then
    printf "%s\n" "${PYTHON}"
    return 0
  fi

  for candidate in python3.9 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      printf "%s\n" "${candidate}"
      return 0
    fi
  done

  return 1
}

PYTHON_CMD="$(find_python || true)"
if [ -z "${PYTHON_CMD}" ]; then
  echo "Python 3.9 is required."
  echo "Set PYTHON=/path/to/python before running this script if it is not on PATH."
  exit 1
fi

if [ -d ".venv" ]; then
  echo "Activate venv..."
  . .venv/bin/activate
else
  echo "Create venv..."
  requirements_file="requirements.txt"

  "${PYTHON_CMD}" -m venv .venv
  . .venv/bin/activate

  if [ -f "${requirements_file}" ]; then
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install "torch==2.8.0" "torchaudio==2.8.0" "torchvision==0.23.0"
    python -m pip install -r "${requirements_file}"
  else
    echo "${requirements_file} not found. Please ensure the requirements file with required packages exists."
    exit 1
  fi
fi

# sounddevice is only listed in the Windows realtime requirements; ensure it here
python -c "import sounddevice" 2>/dev/null || python -m pip install sounddevice

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required for the Electron UI. Install Node.js (https://nodejs.org) first."
  exit 1
fi

if [ ! -d "electron/node_modules" ]; then
  echo "Installing Electron UI dependencies..."
  npm install --prefix electron
fi

echo "Starting the realtime voice conversion UI..."
exec npm start --prefix electron
```

- [ ] **Step 2: Fix the misleading stub in tools/rvc_for_realtime.py**

The module is still imported by `api_231006.py` and `api_240604.py`, so only the `__main__` block changes. Replace:

```python
if __name__ == '__main__':
    initialize_multiprocessing()
    # Real-time VC app initialization would go here
    print("Real-time voice conversion ready")
```

with:

```python
if __name__ == "__main__":
    print(
        "tools/rvc_for_realtime.py is a library used by the REST APIs.\n"
        "For the realtime voice conversion UI, run: ./run-realtime.sh"
    )
```

(`initialize_multiprocessing()` stays — the API entry points reference this module.)

- [ ] **Step 3: Verify the script end-to-end**

```bash
./run-realtime.sh
```

Expected: "Activate venv...", then the Electron window opens and reaches "Ready" as in Task 6 Step 4. Quit with Cmd+Q.

Also verify the stub message:

```bash
.venv/bin/python tools/rvc_for_realtime.py
```

Expected: the two-line pointer message, exit code 0.

- [ ] **Step 4: Commit**

```bash
git add run-realtime.sh tools/rvc_for_realtime.py
git commit -m "launch electron realtime ui from run-realtime.sh

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Final verification

- [ ] **Step 1: Re-run the protocol smoke test**

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python tools/realtime/test_protocol.py
```

Expected: `PROTOCOL SMOKE TEST PASSED`.

- [ ] **Step 2: Check formatting**

The project enforces Black on Python files:

```bash
.venv/bin/python -c "import black" 2>/dev/null || .venv/bin/python -m pip install black
.venv/bin/python -m black tools/realtime/ tools/rvc_for_realtime.py
```

If Black reformatted anything, commit:

```bash
git add -u && git commit -m "format with black

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 3: Manual conversion test (needs the user)**

With a voice model `.pth` + `.index` available: `./run-realtime.sh`, pick the model files, choose input (microphone) and output (headphones) devices, press **Start conversion**, speak, and confirm converted audio plus live latency/inference-time numbers. Hot-tweak pitch while running. Press **Stop**, then quit.

This step requires a voice model — `assets/weights/` is empty today. If the user has none, report the automated results and ask them to test with their model.
