# api for 240604 release version by Xiaokai
import os
import sys
import json
import re
import time
import librosa
import torch
import numpy as np
import torch.nn.functional as F
import torchaudio.transforms as tat
import sounddevice as sd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import threading
import uvicorn
import logging
from multiprocessing import Queue, Process, cpu_count, freeze_support

# Initialize the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define FastAPI app
app = FastAPI()


class GUIConfig:
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
        self.f0method: str = "fcpe"
        self.sg_input_device: str = ""
        self.sg_output_device: str = ""


class ConfigData(BaseModel):
    pth_path: str
    index_path: str
    sg_input_device: str
    sg_output_device: str
    threhold: int = -60
    pitch: int = 0
    formant: float = 0.0
    index_rate: float = 0.3
    rms_mix_rate: float = 0.0
    block_time: float = 0.25
    crossfade_length: float = 0.05
    extra_time: float = 2.5
    n_cpu: int = 4
    I_noise_reduce: bool = False
    O_noise_reduce: bool = False
    use_pv: bool = False
    f0method: str = "fcpe"


class Harvest(Process):
    def __init__(self, inp_q, opt_q):
        super(Harvest, self).__init__()
        self.inp_q = inp_q
        self.opt_q = opt_q

    def run(self):
        import numpy as np
        import pyworld

        while True:
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


class AudioAPI:
    def __init__(self) -> None:
        self.gui_config = GUIConfig()
        self.config = None  # Initialize Config object as None
        self.flag_vc = False
        self.function = "vc"
        self.delay_time = 0
        self.rvc = None  # Initialize RVC object as None
        self.inp_q = None
        self.opt_q = None
        self.n_cpu = min(cpu_count(), 8)

    def initialize_queues(self):
        self.inp_q = Queue()
        self.opt_q = Queue()
        for _ in range(self.n_cpu):
            p = Harvest(self.inp_q, self.opt_q)
            p.daemon = True
            p.start()

    def load(self):
        input_devices, output_devices, _, _ = self.get_devices()
        try:
            with open("configs/config.json", "r", encoding="utf-8") as j:
                data = json.load(j)
                if data["sg_input_device"] not in input_devices:
                    data["sg_input_device"] = input_devices[sd.default.device[0]]
                if data["sg_output_device"] not in output_devices:
                    data["sg_output_device"] = output_devices[sd.default.device[1]]
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            with open("configs/config.json", "w", encoding="utf-8") as j:
                data = {
                    "pth_path": "",
                    "index_path": "",
                    "sg_input_device": input_devices[sd.default.device[0]],
                    "sg_output_device": output_devices[sd.default.device[1]],
                    "threhold": -60,
                    "pitch": 0,
                    "formant": 0.0,
                    "index_rate": 0,
                    "rms_mix_rate": 0,
                    "block_time": 0.25,
                    "crossfade_length": 0.05,
                    "extra_time": 2.5,
                    "n_cpu": 4,
                    "f0method": "fcpe",
                    "use_jit": False,
                    "use_pv": False,
                }
                json.dump(data, j, ensure_ascii=False)
        return data

    def set_values(self, values):
        logger.info(f"Setting values: {values}")
        if not values.pth_path.strip():
            raise HTTPException(status_code=400, detail="Please select a .pth file")
        if not values.index_path.strip():
            raise HTTPException(status_code=400, detail="Please select an index file")
        self.set_devices(values.sg_input_device, values.sg_output_device)
        self.config.use_jit = False
        self.gui_config.pth_path = values.pth_path
        self.gui_config.index_path = values.index_path
        self.gui_config.threhold = values.threhold
        self.gui_config.pitch = values.pitch
        self.gui_config.formant = values.formant
        self.gui_config.block_time = values.block_time
        self.gui_config.crossfade_time = values.crossfade_length
        self.gui_config.extra_time = values.extra_time
        self.gui_config.I_noise_reduce = values.I_noise_reduce
        self.gui_config.O_noise_reduce = values.O_noise_reduce
        self.gui_config.rms_mix_rate = values.rms_mix_rate
        self.gui_config.index_rate = values.index_rate
        self.gui_config.n_cpu = values.n_cpu
        self.gui_config.use_pv = values.use_pv
        self.gui_config.f0method = values.f0method
        return True

    def start_vc(self):
        torch.cuda.empty_cache()
        self.flag_vc = True
        self.rvc = rvc_for_realtime.RVC(
            self.gui_config.pitch,
            self.gui_config.pth_path,
            self.gui_config.index_path,
            self.gui_config.index_rate,
            self.gui_config.n_cpu,
            self.inp_q,
            self.opt_q,
            self.config,
            self.rvc if self.rvc else None,
        )
        self.gui_config.samplerate = (
            self.rvc.tgt_sr
            if self.gui_config.sr_type == "sr_model"
            else self.get_device_samplerate()
        )
        self.zc = self.gui_config.samplerate // 100
        self.block_frame = (
            int(
                np.round(
                    self.gui_config.block_time * self.gui_config.samplerate / self.zc
                )
            )
            * self.zc
        )
        self.block_frame_16k = 160 * self.block_frame // self.zc
        self.crossfade_frame = (
            int(
                np.round(
                    self.gui_config.crossfade_time
                    * self.gui_config.samplerate
                    / self.zc
                )
            )
            * self.zc
        )
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = (
            int(
                np.round(
                    self.gui_config.extra_time * self.gui_config.samplerate / self.zc
                )
            )
            * self.zc
        )
        self.input_wav = torch.zeros(
            self.extra_frame
            + self.crossfade_frame
            + self.sola_search_frame
            + self.block_frame,
            device=self.config.device,
            dtype=torch.float32,
        )
        self.input_wav_denoise = self.input_wav.clone()
        self.input_wav_res = torch.zeros(
            160 * self.input_wav.shape[0] // self.zc,
            device=self.config.device,
            dtype=torch.float32,
        )
        self.rms_buffer = np.zeros(4 * self.zc, dtype="float32")
        self.sola_buffer = torch.zeros(
            self.sola_buffer_frame, device=self.config.device, dtype=torch.float32
        )
        self.nr_buffer = self.sola_buffer.clone()
        self.output_buffer = self.input_wav.clone()
        self.skip_head = self.extra_frame // self.zc
        self.return_length = (
            self.block_frame + self.sola_buffer_frame + self.sola_search_frame
        ) // self.zc
        self.fade_in_window = (
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
        self.fade_out_window = 1 - self.fade_in_window
        self.resampler = tat.Resample(
            orig_freq=self.gui_config.samplerate,
            new_freq=16000,
            dtype=torch.float32,
        ).to(self.config.device)
        if self.rvc.tgt_sr != self.gui_config.samplerate:
            self.resampler2 = tat.Resample(
                orig_freq=self.rvc.tgt_sr,
                new_freq=self.gui_config.samplerate,
                dtype=torch.float32,
            ).to(self.config.device)
        else:
            self.resampler2 = None
        self.tg = TorchGate(
            sr=self.gui_config.samplerate, n_fft=4 * self.zc, prop_decrease=0.9
        ).to(self.config.device)
        thread_vc = threading.Thread(target=self.soundinput)
        thread_vc.start()

    def soundinput(self):
        channels = 1 if sys.platform == "darwin" else 2
        with sd.Stream(
            channels=channels,
            callback=self.audio_callback,
            blocksize=self.block_frame,
            samplerate=self.gui_config.samplerate,
            dtype="float32",
        ) as stream:
            global stream_latency
            stream_latency = stream.latency[-1]
            while self.flag_vc:
                time.sleep(self.gui_config.block_time)
                logger.info("Audio block passed.")
        logger.info("Ending VC")

    def audio_callback(
        self, indata: np.ndarray, outdata: np.ndarray, frames, times, status
    ):
        start_time = time.perf_counter()
        indata = librosa.to_mono(indata.T)
        if self.gui_config.threhold > -60:
            indata = np.append(self.rms_buffer, indata)
            rms = librosa.feature.rms(
                y=indata, frame_length=4 * self.zc, hop_length=self.zc
            )[:, 2:]
            self.rms_buffer[:] = indata[-4 * self.zc :]
            indata = indata[2 * self.zc - self.zc // 2 :]
            db_threhold = (
                librosa.amplitude_to_db(rms, ref=1.0)[0] < self.gui_config.threhold
            )
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
        if self.gui_config.I_noise_reduce:
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
                self.gui_config.f0method,
            )
            if self.resampler2 is not None:
                infer_wav = self.resampler2(infer_wav)
        elif self.gui_config.I_noise_reduce:
            infer_wav = self.input_wav_denoise[self.extra_frame :].clone()
        else:
            infer_wav = self.input_wav[self.extra_frame :].clone()
        # output noise reduction
        if self.gui_config.O_noise_reduce and self.function == "vc":
            self.output_buffer[: -self.block_frame] = self.output_buffer[
                self.block_frame :
            ].clone()
            self.output_buffer[-self.block_frame :] = infer_wav[-self.block_frame :]
            infer_wav = self.tg(
                infer_wav.unsqueeze(0), self.output_buffer.unsqueeze(0)
            ).squeeze(0)
        # volume envelop mixing
        if self.gui_config.rms_mix_rate < 1 and self.function == "vc":
            if self.gui_config.I_noise_reduce:
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
                rms1 / rms2, torch.tensor(1 - self.gui_config.rms_mix_rate)
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
        logger.info(f"sola_offset = {sola_offset}")
        infer_wav = infer_wav[sola_offset:]
        if "privateuseone" in str(self.config.device) or not self.gui_config.use_pv:
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
        if sys.platform == "darwin":
            outdata[:] = infer_wav[: self.block_frame].cpu().numpy()[:, np.newaxis]
        else:
            outdata[:] = infer_wav[: self.block_frame].repeat(2, 1).t().cpu().numpy()
        total_time = time.perf_counter() - start_time
        logger.info(f"Infer time: {total_time:.2f}")

    def get_devices(self, update: bool = True):
        if update:
            sd._terminate()
            sd._initialize()
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        for hostapi in hostapis:
            for device_idx in hostapi["devices"]:
                devices[device_idx]["hostapi_name"] = hostapi["name"]
        input_devices = [
            f"{d['name']} ({d['hostapi_name']})"
            for d in devices
            if d["max_input_channels"] > 0
        ]
        output_devices = [
            f"{d['name']} ({d['hostapi_name']})"
            for d in devices
            if d["max_output_channels"] > 0
        ]
        input_devices_indices = [
            d["index"] if "index" in d else d["name"]
            for d in devices
            if d["max_input_channels"] > 0
        ]
        output_devices_indices = [
            d["index"] if "index" in d else d["name"]
            for d in devices
            if d["max_output_channels"] > 0
        ]
        return (
            input_devices,
            output_devices,
            input_devices_indices,
            output_devices_indices,
        )

    def set_devices(self, input_device, output_device):
        (
            input_devices,
            output_devices,
            input_device_indices,
            output_device_indices,
        ) = self.get_devices()
        logger.debug(f"Available input devices: {input_devices}")
        logger.debug(f"Available output devices: {output_devices}")
        logger.debug(f"Selected input device: {input_device}")
        logger.debug(f"Selected output device: {output_device}")

        if input_device not in input_devices:
            logger.error(
                f"Input device '{input_device}' is not in the list of available devices"
            )
            raise HTTPException(
                status_code=400,
                detail=f"Input device '{input_device}' is not available",
            )

        if output_device not in output_devices:
            logger.error(
                f"Output device '{output_device}' is not in the list of available devices"
            )
            raise HTTPException(
                status_code=400,
                detail=f"Output device '{output_device}' is not available",
            )

        sd.default.device[0] = input_device_indices[input_devices.index(input_device)]
        sd.default.device[1] = output_device_indices[
            output_devices.index(output_device)
        ]
        logger.info(f"Input device set to {sd.default.device[0]}: {input_device}")
        logger.info(f"Output device set to {sd.default.device[1]}: {output_device}")


audio_api = AudioAPI()


@app.get("/inputDevices", response_model=list)
def get_input_devices():
    try:
        input_devices, _, _, _ = audio_api.get_devices()
        return input_devices
    except Exception as e:
        logger.error(f"Failed to get input devices: {e}")
        raise HTTPException(status_code=500, detail="Failed to get input devices")


@app.get("/outputDevices", response_model=list)
def get_output_devices():
    try:
        _, output_devices, _, _ = audio_api.get_devices()
        return output_devices
    except Exception as e:
        logger.error(f"Failed to get output devices: {e}")
        raise HTTPException(status_code=500, detail="Failed to get output devices")


@app.post("/config")
def configure_audio(config_data: ConfigData):
    try:
        logger.info(f"Configuring audio with data: {config_data}")
        if audio_api.set_values(config_data):
            settings = config_data.dict()
            settings["use_jit"] = False
            with open("configs/config.json", "w", encoding="utf-8") as j:
                json.dump(settings, j, ensure_ascii=False)
            logger.info("Configuration set successfully")
            return {"message": "Configuration set successfully"}
    except HTTPException as e:
        logger.error(f"Configuration error: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Configuration failed: {e}")
        raise HTTPException(status_code=400, detail=f"Configuration failed: {e}")


@app.post("/start")
def start_conversion():
    try:
        if not audio_api.flag_vc:
            audio_api.start_vc()
            return {"message": "Audio conversion started"}
        else:
            logger.warning("Audio conversion already running")
            raise HTTPException(
                status_code=400, detail="Audio conversion already running"
            )
    except HTTPException as e:
        logger.error(f"Start conversion error: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Failed to start conversion: {e}")
        raise HTTPException(status_code=500, detail="Failed to start conversion: {e}")


@app.post("/stop")
def stop_conversion():
    try:
        if audio_api.flag_vc:
            audio_api.flag_vc = False
            global stream_latency
            stream_latency = -1
            return {"message": "Audio conversion stopped"}
        else:
            logger.warning("Audio conversion not running")
            raise HTTPException(status_code=400, detail="Audio conversion not running")
    except HTTPException as e:
        logger.error(f"Stop conversion error: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Failed to stop conversion: {e}")
        raise HTTPException(status_code=500, detail="Failed to stop conversion: {e}")


if __name__ == "__main__":
    if sys.platform == "win32":
        freeze_support()
    load_dotenv()
    os.environ["OMP_NUM_THREADS"] = "4"
    if sys.platform == "darwin":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    from tools.torchgate import TorchGate
    import tools.rvc_for_realtime as rvc_for_realtime
    from configs.config import Config

    audio_api.config = Config()
    audio_api.initialize_queues()
    uvicorn.run(app, host="0.0.0.0", port=6242)
