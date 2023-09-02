import os
import logging
import sys

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["rmvpe_root"] = "assets/rmvpe"
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)
import multiprocessing

logger = logging.getLogger(__name__)


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


if __name__ == "__main__":
    import json
    import multiprocessing
    import re
    import threading
    import time
    import traceback
    from multiprocessing import Queue, cpu_count
    from queue import Empty

    import librosa
    import noisereduce as nr
    import numpy as np
    import PySimpleGUI as sg
    import sounddevice as sd
    import torch
    import torch.nn.functional as F
    import torchaudio.transforms as tat

    import tools.rvc_for_realtime as rvc_for_realtime
    from i18n.i18n import I18nAuto

    i18n = I18nAuto()
    device = rvc_for_realtime.config.device
    # device = torch.device(
    #     "cuda"
    #     if torch.cuda.is_available()
    #     else ("mps" if torch.backends.mps.is_available() else "cpu")
    # )
    current_dir = os.getcwd()
    inp_q = Queue()
    opt_q = Queue()
    n_cpu = min(cpu_count(), 8)
    for _ in range(n_cpu):
        Harvest(inp_q, opt_q).start()

    class GUIConfig:
        def __init__(self) -> None:
            self.pth_path: str = ""
            self.index_path: str = ""
            self.pitch: int = 12
            self.samplerate: int = 40000
            self.block_time: float = 1.0  # s
            self.buffer_num: int = 1
            self.threhold: int = -30
            self.crossfade_time: float = 0.08
            self.extra_time: float = 0.04
            self.I_noise_reduce = False
            self.O_noise_reduce = False
            self.index_rate = 0.3
            self.n_cpu = min(n_cpu, 6)
            self.f0method = "harvest"
            self.sg_input_device = ""
            self.sg_output_device = ""

    class GUI:
        def __init__(self) -> None:
            self.config = GUIConfig()
            self.flag_vc = False

            self.launcher()

        def load(self):
            input_devices, output_devices, _, _ = self.get_devices()
            try:
                with open("configs/config.json", "r") as j:
                    data = json.load(j)
                    data["pm"] = data["f0method"] == "pm"
                    data["harvest"] = data["f0method"] == "harvest"
                    data["crepe"] = data["f0method"] == "crepe"
                    data["rmvpe"] = data["f0method"] == "rmvpe"
            except:
                with open("configs/config.json", "w") as j:
                    data = {
                        "pth_path": " ",
                        "index_path": " ",
                        "sg_input_device": input_devices[sd.default.device[0]],
                        "sg_output_device": output_devices[sd.default.device[1]],
                        "threhold": "-45",
                        "pitch": "0",
                        "index_rate": "0",
                        "block_time": "1",
                        "crossfade_length": "0.04",
                        "extra_time": "1",
                        "f0method": "rmvpe",
                    }
            return data

        def launcher(self):
            data = self.load()
            sg.theme("LightBlue3")
            input_devices, output_devices, _, _ = self.get_devices()
            layout = [
                [
                    sg.Frame(
                        title=i18n("加载模型"),
                        layout=[
                            [
                                sg.Input(
                                    default_text=data.get("pth_path", ""),
                                    key="pth_path",
                                ),
                                sg.FileBrowse(
                                    i18n("选择.pth文件"),
                                    initial_folder=os.path.join(
                                        os.getcwd(), "assets/weights"
                                    ),
                                    file_types=((". pth"),),
                                ),
                            ],
                            [
                                sg.Input(
                                    default_text=data.get("index_path", ""),
                                    key="index_path",
                                ),
                                sg.FileBrowse(
                                    i18n("选择.index文件"),
                                    initial_folder=os.path.join(os.getcwd(), "logs"),
                                    file_types=((". index"),),
                                ),
                            ],
                        ],
                    )
                ],
                [
                    sg.Frame(
                        layout=[
                            [
                                sg.Text(i18n("输入设备")),
                                sg.Combo(
                                    input_devices,
                                    key="sg_input_device",
                                    default_value=data.get("sg_input_device", ""),
                                ),
                            ],
                            [
                                sg.Text(i18n("输出设备")),
                                sg.Combo(
                                    output_devices,
                                    key="sg_output_device",
                                    default_value=data.get("sg_output_device", ""),
                                ),
                            ],
                            [sg.Button(i18n("重载设备列表"), key="reload_devices")],
                        ],
                        title=i18n("音频设备(请使用同种类驱动)"),
                    )
                ],
                [
                    sg.Frame(
                        layout=[
                            [
                                sg.Text(i18n("响应阈值")),
                                sg.Slider(
                                    range=(-60, 0),
                                    key="threhold",
                                    resolution=1,
                                    orientation="h",
                                    default_value=data.get("threhold", ""),
                                    enable_events=True,
                                ),
                            ],
                            [
                                sg.Text(i18n("音调设置")),
                                sg.Slider(
                                    range=(-24, 24),
                                    key="pitch",
                                    resolution=1,
                                    orientation="h",
                                    default_value=data.get("pitch", ""),
                                    enable_events=True,
                                ),
                            ],
                            [
                                sg.Text(i18n("Index Rate")),
                                sg.Slider(
                                    range=(0.0, 1.0),
                                    key="index_rate",
                                    resolution=0.01,
                                    orientation="h",
                                    default_value=data.get("index_rate", ""),
                                    enable_events=True,
                                ),
                            ],
                            [
                                sg.Text(i18n("音高算法")),
                                sg.Radio(
                                    "pm",
                                    "f0method",
                                    key="pm",
                                    default=data.get("pm", "") == True,
                                    enable_events=True,
                                ),
                                sg.Radio(
                                    "harvest",
                                    "f0method",
                                    key="harvest",
                                    default=data.get("harvest", "") == True,
                                    enable_events=True,
                                ),
                                sg.Radio(
                                    "crepe",
                                    "f0method",
                                    key="crepe",
                                    default=data.get("crepe", "") == True,
                                    enable_events=True,
                                ),
                                sg.Radio(
                                    "rmvpe",
                                    "f0method",
                                    key="rmvpe",
                                    default=data.get("rmvpe", "") == True,
                                    enable_events=True,
                                ),
                            ],
                        ],
                        title=i18n("常规设置"),
                    ),
                    sg.Frame(
                        layout=[
                            [
                                sg.Text(i18n("采样长度")),
                                sg.Slider(
                                    range=(0.05, 2.4),
                                    key="block_time",
                                    resolution=0.01,
                                    orientation="h",
                                    default_value=data.get("block_time", ""),
                                    enable_events=True,
                                ),
                            ],
                            [
                                sg.Text(i18n("harvest进程数")),
                                sg.Slider(
                                    range=(1, n_cpu),
                                    key="n_cpu",
                                    resolution=1,
                                    orientation="h",
                                    default_value=data.get(
                                        "n_cpu", min(self.config.n_cpu, n_cpu)
                                    ),
                                    enable_events=True,
                                ),
                            ],
                            [
                                sg.Text(i18n("淡入淡出长度")),
                                sg.Slider(
                                    range=(0.01, 0.15),
                                    key="crossfade_length",
                                    resolution=0.01,
                                    orientation="h",
                                    default_value=data.get("crossfade_length", ""),
                                    enable_events=True,
                                ),
                            ],
                            [
                                sg.Text(i18n("额外推理时长")),
                                sg.Slider(
                                    range=(0.05, 5.00),
                                    key="extra_time",
                                    resolution=0.01,
                                    orientation="h",
                                    default_value=data.get("extra_time", ""),
                                    enable_events=True,
                                ),
                            ],
                            [
                                sg.Checkbox(
                                    i18n("输入降噪"),
                                    key="I_noise_reduce",
                                    enable_events=True,
                                ),
                                sg.Checkbox(
                                    i18n("输出降噪"),
                                    key="O_noise_reduce",
                                    enable_events=True,
                                ),
                            ],
                        ],
                        title=i18n("性能设置"),
                    ),
                ],
                [
                    sg.Button(i18n("开始音频转换"), key="start_vc"),
                    sg.Button(i18n("停止音频转换"), key="stop_vc"),
                    sg.Text(i18n("推理时间(ms):")),
                    sg.Text("0", key="infer_time"),
                ],
            ]
            self.window = sg.Window("RVC - GUI", layout=layout, finalize=True)
            self.event_handler()

        def event_handler(self):
            while True:
                event, values = self.window.read()
                if event == sg.WINDOW_CLOSED:
                    self.flag_vc = False
                    exit()
                if event == "reload_devices":
                    prev_input = self.window["sg_input_device"].get()
                    prev_output = self.window["sg_output_device"].get()
                    input_devices, output_devices, _, _ = self.get_devices(update=True)
                    if prev_input not in input_devices:
                        self.config.sg_input_device = input_devices[0]
                    else:
                        self.config.sg_input_device = prev_input
                    self.window["sg_input_device"].Update(values=input_devices)
                    self.window["sg_input_device"].Update(
                        value=self.config.sg_input_device
                    )
                    if prev_output not in output_devices:
                        self.config.sg_output_device = output_devices[0]
                    else:
                        self.config.sg_output_device = prev_output
                    self.window["sg_output_device"].Update(values=output_devices)
                    self.window["sg_output_device"].Update(
                        value=self.config.sg_output_device
                    )
                if event == "start_vc" and self.flag_vc == False:
                    if self.set_values(values) == True:
                        logger.info("Use CUDA: %s", torch.cuda.is_available())
                        self.start_vc()
                        settings = {
                            "pth_path": values["pth_path"],
                            "index_path": values["index_path"],
                            "sg_input_device": values["sg_input_device"],
                            "sg_output_device": values["sg_output_device"],
                            "threhold": values["threhold"],
                            "pitch": values["pitch"],
                            "index_rate": values["index_rate"],
                            "block_time": values["block_time"],
                            "crossfade_length": values["crossfade_length"],
                            "extra_time": values["extra_time"],
                            "n_cpu": values["n_cpu"],
                            "f0method": ["pm", "harvest", "crepe", "rmvpe"][
                                [
                                    values["pm"],
                                    values["harvest"],
                                    values["crepe"],
                                    values["rmvpe"],
                                ].index(True)
                            ],
                        }
                        with open("configs/config.json", "w") as j:
                            json.dump(settings, j)
                if event == "stop_vc" and self.flag_vc == True:
                    self.flag_vc = False

                # Parameter hot update
                if event == "threhold":
                    self.config.threhold = values["threhold"]
                elif event == "pitch":
                    self.config.pitch = values["pitch"]
                    if hasattr(self, "rvc"):
                        self.rvc.change_key(values["pitch"])
                elif event == "index_rate":
                    self.config.index_rate = values["index_rate"]
                    if hasattr(self, "rvc"):
                        self.rvc.change_index_rate(values["index_rate"])
                elif event in ["pm", "harvest", "crepe", "rmvpe"]:
                    self.config.f0method = event
                elif event == "I_noise_reduce":
                    self.config.I_noise_reduce = values["I_noise_reduce"]
                elif event == "O_noise_reduce":
                    self.config.O_noise_reduce = values["O_noise_reduce"]
                elif event != "start_vc" and self.flag_vc == True:
                    # Other parameters do not support hot update
                    self.flag_vc = False

        def set_values(self, values):
            if len(values["pth_path"].strip()) == 0:
                sg.popup(i18n("请选择pth文件"))
                return False
            if len(values["index_path"].strip()) == 0:
                sg.popup(i18n("请选择index文件"))
                return False
            pattern = re.compile("[^\x00-\x7F]+")
            if pattern.findall(values["pth_path"]):
                sg.popup(i18n("pth文件路径不可包含中文"))
                return False
            if pattern.findall(values["index_path"]):
                sg.popup(i18n("index文件路径不可包含中文"))
                return False
            self.set_devices(values["sg_input_device"], values["sg_output_device"])
            self.config.pth_path = values["pth_path"]
            self.config.index_path = values["index_path"]
            self.config.threhold = values["threhold"]
            self.config.pitch = values["pitch"]
            self.config.block_time = values["block_time"]
            self.config.crossfade_time = values["crossfade_length"]
            self.config.extra_time = values["extra_time"]
            self.config.I_noise_reduce = values["I_noise_reduce"]
            self.config.O_noise_reduce = values["O_noise_reduce"]
            self.config.index_rate = values["index_rate"]
            self.config.n_cpu = values["n_cpu"]
            self.config.f0method = ["pm", "harvest", "crepe", "rmvpe"][
                [
                    values["pm"],
                    values["harvest"],
                    values["crepe"],
                    values["rmvpe"],
                ].index(True)
            ]
            return True

        def start_vc(self):
            torch.cuda.empty_cache()
            self.flag_vc = True
            self.rvc = rvc_for_realtime.RVC(
                self.config.pitch,
                self.config.pth_path,
                self.config.index_path,
                self.config.index_rate,
                self.config.n_cpu,
                inp_q,
                opt_q,
                device,
                self.rvc if hasattr(self, "rvc") else None,
            )
            self.config.samplerate = self.rvc.tgt_sr
            self.config.crossfade_time = min(
                self.config.crossfade_time, self.config.block_time
            )
            self.zc = self.rvc.tgt_sr // 100
            self.block_frame = (
                int(np.round(self.config.block_time * self.config.samplerate / self.zc))
                * self.zc
            )
            self.block_frame_16k = 160 * self.block_frame // self.zc
            self.crossfade_frame = int(
                self.config.crossfade_time * self.config.samplerate
            )
            self.sola_search_frame = int(0.01 * self.config.samplerate)
            self.extra_frame = int(self.config.extra_time * self.config.samplerate)
            self.input_wav: np.ndarray = np.zeros(
                int(
                    np.ceil(
                        (
                            self.extra_frame
                            + self.crossfade_frame
                            + self.sola_search_frame
                            + self.block_frame
                        )
                        / self.zc
                    )
                    * self.zc
                ),
                dtype="float32",
            )
            self.input_wav_res: torch.Tensor = torch.zeros(
                160 * len(self.input_wav) // self.zc
            )
            self.output_wav_cache: torch.Tensor = torch.zeros(
                int(
                    np.ceil(
                        (
                            self.extra_frame
                            + self.crossfade_frame
                            + self.sola_search_frame
                            + self.block_frame
                        )
                        / self.zc
                    )
                    * self.zc
                ),
                device=device,
                dtype=torch.float32,
            )
            self.pitch: np.ndarray = np.zeros(
                self.input_wav.shape[0] // self.zc,
                dtype="int32",
            )
            self.pitchf: np.ndarray = np.zeros(
                self.input_wav.shape[0] // self.zc,
                dtype="float64",
            )
            self.output_wav: torch.Tensor = torch.zeros(
                self.block_frame, device=device, dtype=torch.float32
            )
            self.sola_buffer: torch.Tensor = torch.zeros(
                self.crossfade_frame, device=device, dtype=torch.float32
            )
            self.fade_in_window: torch.Tensor = (
                torch.sin(
                    0.5
                    * np.pi
                    * torch.linspace(
                        0.0,
                        1.0,
                        steps=self.crossfade_frame,
                        device=device,
                        dtype=torch.float32,
                    )
                )
                ** 2
            )
            self.fade_out_window: torch.Tensor = 1 - self.fade_in_window
            self.resampler = tat.Resample(
                orig_freq=self.config.samplerate, new_freq=16000, dtype=torch.float32
            ).to(device)
            thread_vc = threading.Thread(target=self.soundinput)
            thread_vc.start()

        def soundinput(self):
            """
            接受音频输入
            """
            channels = 1 if sys.platform == "darwin" else 2
            with sd.Stream(
                channels=channels,
                callback=self.audio_callback,
                blocksize=self.block_frame,
                samplerate=self.config.samplerate,
                dtype="float32",
            ):
                while self.flag_vc:
                    time.sleep(self.config.block_time)
                    logger.debug("Audio block passed.")
            logger.debug("ENDing VC")

        def audio_callback(
            self, indata: np.ndarray, outdata: np.ndarray, frames, times, status
        ):
            """
            音频处理
            """
            start_time = time.perf_counter()
            indata = librosa.to_mono(indata.T)
            if self.config.I_noise_reduce:
                indata[:] = nr.reduce_noise(y=indata, sr=self.config.samplerate)
            """noise gate"""
            frame_length = 2048
            hop_length = 1024
            rms = librosa.feature.rms(
                y=indata, frame_length=frame_length, hop_length=hop_length
            )
            if self.config.threhold > -60:
                db_threhold = (
                    librosa.amplitude_to_db(rms, ref=1.0)[0] < self.config.threhold
                )
                for i in range(db_threhold.shape[0]):
                    if db_threhold[i]:
                        indata[i * hop_length : (i + 1) * hop_length] = 0
            self.input_wav[: -self.block_frame] = self.input_wav[self.block_frame :]
            self.input_wav[-self.block_frame :] = indata

            # infer
            inp = torch.from_numpy(
                self.input_wav[-self.block_frame - 2 * self.zc :]
            ).to(device)
            self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[
                self.block_frame_16k :
            ].clone()
            self.input_wav_res[-self.block_frame_16k - 160 :] = self.resampler(inp)[
                160:
            ]
            rate = (
                self.crossfade_frame + self.sola_search_frame + self.block_frame
            ) / (
                self.extra_frame
                + self.crossfade_frame
                + self.sola_search_frame
                + self.block_frame
            )
            f0_extractor_frame = self.block_frame_16k + 800
            if self.config.f0method == "rmvpe":
                f0_extractor_frame = 5120 * ((f0_extractor_frame - 1) // 5120 + 1)
            res2 = self.rvc.infer(
                self.input_wav_res,
                self.input_wav_res[-f0_extractor_frame:].cpu().numpy(),
                self.block_frame_16k,
                rate,
                self.pitch,
                self.pitchf,
                self.config.f0method,
            )
            self.output_wav_cache[-res2.shape[0] :] = res2
            infer_wav = self.output_wav_cache[
                -self.crossfade_frame - self.sola_search_frame - self.block_frame :
            ]
            # SOLA algorithm from https://github.com/yxlllc/DDSP-SVC
            cor_nom = F.conv1d(
                infer_wav[None, None, : self.crossfade_frame + self.sola_search_frame],
                self.sola_buffer[None, None, :],
            )
            cor_den = torch.sqrt(
                F.conv1d(
                    infer_wav[
                        None, None, : self.crossfade_frame + self.sola_search_frame
                    ]
                    ** 2,
                    torch.ones(1, 1, self.crossfade_frame, device=device),
                )
                + 1e-8
            )
            if sys.platform == "darwin":
                _, sola_offset = torch.max(cor_nom[0, 0] / cor_den[0, 0])
                sola_offset = sola_offset.item()
            else:
                sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
            logger.debug("sola_offset = %d", int(sola_offset))
            self.output_wav[:] = infer_wav[sola_offset : sola_offset + self.block_frame]
            self.output_wav[: self.crossfade_frame] *= self.fade_in_window
            self.output_wav[: self.crossfade_frame] += self.sola_buffer[:]
            # crossfade
            if sola_offset < self.sola_search_frame:
                self.sola_buffer[:] = (
                    infer_wav[
                        -self.sola_search_frame
                        - self.crossfade_frame
                        + sola_offset : -self.sola_search_frame
                        + sola_offset
                    ]
                    * self.fade_out_window
                )
            else:
                self.sola_buffer[:] = (
                    infer_wav[-self.crossfade_frame :] * self.fade_out_window
                )
            if self.config.O_noise_reduce:
                if sys.platform == "darwin":
                    noise_reduced_signal = nr.reduce_noise(
                        y=self.output_wav[:].cpu().numpy(), sr=self.config.samplerate
                    )
                    outdata[:] = noise_reduced_signal[:, np.newaxis]
                else:
                    outdata[:] = np.tile(
                        nr.reduce_noise(
                            y=self.output_wav[:].cpu().numpy(),
                            sr=self.config.samplerate,
                        ),
                        (2, 1),
                    ).T
            else:
                if sys.platform == "darwin":
                    outdata[:] = self.output_wav[:].cpu().numpy()[:, np.newaxis]
                else:
                    outdata[:] = self.output_wav[:].repeat(2, 1).t().cpu().numpy()
            total_time = time.perf_counter() - start_time
            self.window["infer_time"].update(int(total_time * 1000))
            logger.info("Infer time: %.2f", total_time)

        def get_devices(self, update: bool = True):
            """获取设备列表"""
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
            """设置输出设备"""
            (
                input_devices,
                output_devices,
                input_device_indices,
                output_device_indices,
            ) = self.get_devices()
            sd.default.device[0] = input_device_indices[
                input_devices.index(input_device)
            ]
            sd.default.device[1] = output_device_indices[
                output_devices.index(output_device)
            ]
            logger.info("Input device: %s:%d", str(sd.default.device[0]), input_device)
            logger.info(
                "Output device: %s:%d", str(sd.default.device[1]), output_device
            )

    gui = GUI()
