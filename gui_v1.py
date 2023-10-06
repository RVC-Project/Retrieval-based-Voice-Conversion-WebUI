import os
import sys
from dotenv import load_dotenv

load_dotenv()

os.environ["OMP_NUM_THREADS"] = "4"
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)
import multiprocessing

stream_latency = -1


def printt(strr, *args):
    if len(args) == 0:
        print(strr)
    else:
        print(strr % args)


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
    from tools.torchgate import TorchGate
    import numpy as np
    import PySimpleGUI as sg
    import sounddevice as sd
    import torch
    import torch.nn.functional as F
    import torchaudio.transforms as tat

    import tools.rvc_for_realtime as rvc_for_realtime
    from i18n.i18n import I18nAuto
    from configs.config import Config

    i18n = I18nAuto()

    # device = rvc_for_realtime.config.device
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
            self.pitch: int = 0
            self.samplerate: int = 40000
            self.block_time: float = 1.0  # s
            self.buffer_num: int = 1
            self.threhold: int = -60
            self.crossfade_time: float = 0.05
            self.extra_time: float = 2.5
            self.I_noise_reduce = False
            self.O_noise_reduce = False
            self.rms_mix_rate = 0.0
            self.index_rate = 0.3
            self.n_cpu = min(n_cpu, 6)
            self.f0method = "harvest"
            self.sg_input_device = ""
            self.sg_output_device = ""

    class GUI:
        def __init__(self) -> None:
            self.gui_config = GUIConfig()
            self.config = Config()
            self.flag_vc = False
            self.function = "vc"
            self.delay_time = 0
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
                    if data["sg_input_device"] not in input_devices:
                        data["sg_input_device"] = input_devices[sd.default.device[0]]
                    if data["sg_output_device"] not in output_devices:
                        data["sg_output_device"] = output_devices[sd.default.device[1]]
            except:
                with open("configs/config.json", "w") as j:
                    data = {
                        "pth_path": " ",
                        "index_path": " ",
                        "sg_input_device": input_devices[sd.default.device[0]],
                        "sg_output_device": output_devices[sd.default.device[1]],
                        "threhold": "-60",
                        "pitch": "0",
                        "index_rate": "0",
                        "rms_mix_rate": "0",
                        "block_time": "0.25",
                        "crossfade_length": "0.05",
                        "extra_time": "2.5",
                        "f0method": "rmvpe",
                        "use_jit": False,
                    }
                    data["pm"] = data["f0method"] == "pm"
                    data["harvest"] = data["f0method"] == "harvest"
                    data["crepe"] = data["f0method"] == "crepe"
                    data["rmvpe"] = data["f0method"] == "rmvpe"
            return data

        def launcher(self):
            data = self.load()
            self.config.use_jit = False  # data.get("use_jit", self.config.use_jit)
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
                                    default_value=data.get("threhold", "-60"),
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
                                    default_value=data.get("pitch", "0"),
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
                                    default_value=data.get("index_rate", "0"),
                                    enable_events=True,
                                ),
                            ],
                            [
                                sg.Text(i18n("响度因子")),
                                sg.Slider(
                                    range=(0.0, 1.0),
                                    key="rms_mix_rate",
                                    resolution=0.01,
                                    orientation="h",
                                    default_value=data.get("rms_mix_rate", "0"),
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
                                    default_value=data.get("block_time", "0.25"),
                                    enable_events=True,
                                ),
                            ],
                            # [
                            #     sg.Text("设备延迟"),
                            #     sg.Slider(
                            #         range=(0, 1),
                            #         key="device_latency",
                            #         resolution=0.001,
                            #         orientation="h",
                            #         default_value=data.get("device_latency", "0.1"),
                            #         enable_events=True,
                            #     ),
                            # ],
                            [
                                sg.Text(i18n("harvest进程数")),
                                sg.Slider(
                                    range=(1, n_cpu),
                                    key="n_cpu",
                                    resolution=1,
                                    orientation="h",
                                    default_value=data.get(
                                        "n_cpu", min(self.gui_config.n_cpu, n_cpu)
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
                                    default_value=data.get("crossfade_length", "0.05"),
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
                                    default_value=data.get("extra_time", "2.5"),
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
                                # sg.Checkbox(
                                #     "JIT加速",
                                #     default=self.config.use_jit,
                                #     key="use_jit",
                                #     enable_events=False,
                                # ),
                            ],
                            # [sg.Text("注：首次使用JIT加速时，会出现卡顿，\n      并伴随一些噪音，但这是正常现象！")],
                        ],
                        title=i18n("性能设置"),
                    ),
                ],
                [
                    sg.Button(i18n("开始音频转换"), key="start_vc"),
                    sg.Button(i18n("停止音频转换"), key="stop_vc"),
                    sg.Radio(
                        i18n("输入监听"),
                        "function",
                        key="im",
                        default=False,
                        enable_events=True,
                    ),
                    sg.Radio(
                        i18n("输出变声"),
                        "function",
                        key="vc",
                        default=True,
                        enable_events=True,
                    ),
                    sg.Text(i18n("算法延迟(ms):")),
                    sg.Text("0", key="delay_time"),
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
                        self.gui_config.sg_input_device = input_devices[0]
                    else:
                        self.gui_config.sg_input_device = prev_input
                    self.window["sg_input_device"].Update(values=input_devices)
                    self.window["sg_input_device"].Update(
                        value=self.gui_config.sg_input_device
                    )
                    if prev_output not in output_devices:
                        self.gui_config.sg_output_device = output_devices[0]
                    else:
                        self.gui_config.sg_output_device = prev_output
                    self.window["sg_output_device"].Update(values=output_devices)
                    self.window["sg_output_device"].Update(
                        value=self.gui_config.sg_output_device
                    )
                if event == "start_vc" and self.flag_vc == False:
                    if self.set_values(values) == True:
                        printt("cuda_is_available: %s", torch.cuda.is_available())
                        self.start_vc()
                        settings = {
                            "pth_path": values["pth_path"],
                            "index_path": values["index_path"],
                            "sg_input_device": values["sg_input_device"],
                            "sg_output_device": values["sg_output_device"],
                            "threhold": values["threhold"],
                            "pitch": values["pitch"],
                            "rms_mix_rate": values["rms_mix_rate"],
                            "index_rate": values["index_rate"],
                            # "device_latency": values["device_latency"],
                            "block_time": values["block_time"],
                            "crossfade_length": values["crossfade_length"],
                            "extra_time": values["extra_time"],
                            "n_cpu": values["n_cpu"],
                            # "use_jit": values["use_jit"],
                            "use_jit": False,
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
                        global stream_latency
                        while stream_latency < 0:
                            time.sleep(0.01)
                        self.delay_time = (
                            stream_latency
                            + values["block_time"]
                            + values["crossfade_length"]
                            + 0.01
                        )
                        if values["I_noise_reduce"]:
                            self.delay_time += values["crossfade_length"]
                        self.window["delay_time"].update(int(self.delay_time * 1000))
                if event == "stop_vc" and self.flag_vc == True:
                    self.flag_vc = False
                    stream_latency = -1
                # Parameter hot update
                if event == "threhold":
                    self.gui_config.threhold = values["threhold"]
                elif event == "pitch":
                    self.gui_config.pitch = values["pitch"]
                    if hasattr(self, "rvc"):
                        self.rvc.change_key(values["pitch"])
                elif event == "index_rate":
                    self.gui_config.index_rate = values["index_rate"]
                    if hasattr(self, "rvc"):
                        self.rvc.change_index_rate(values["index_rate"])
                elif event == "rms_mix_rate":
                    self.gui_config.rms_mix_rate = values["rms_mix_rate"]
                elif event in ["pm", "harvest", "crepe", "rmvpe"]:
                    self.gui_config.f0method = event
                elif event == "I_noise_reduce":
                    self.gui_config.I_noise_reduce = values["I_noise_reduce"]
                    if stream_latency > 0:
                        self.delay_time += (
                            1 if values["I_noise_reduce"] else -1
                        ) * values["crossfade_length"]
                        self.window["delay_time"].update(int(self.delay_time * 1000))
                elif event == "O_noise_reduce":
                    self.gui_config.O_noise_reduce = values["O_noise_reduce"]
                elif event in ["vc", "im"]:
                    self.function = event
                elif event != "start_vc" and self.flag_vc == True:
                    # Other parameters do not support hot update
                    self.flag_vc = False
                    stream_latency = -1

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
            self.config.use_jit = False  # values["use_jit"]
            # self.device_latency = values["device_latency"]
            self.gui_config.pth_path = values["pth_path"]
            self.gui_config.index_path = values["index_path"]
            self.gui_config.threhold = values["threhold"]
            self.gui_config.pitch = values["pitch"]
            self.gui_config.block_time = values["block_time"]
            self.gui_config.crossfade_time = values["crossfade_length"]
            self.gui_config.extra_time = values["extra_time"]
            self.gui_config.I_noise_reduce = values["I_noise_reduce"]
            self.gui_config.O_noise_reduce = values["O_noise_reduce"]
            self.gui_config.rms_mix_rate = values["rms_mix_rate"]
            self.gui_config.index_rate = values["index_rate"]
            self.gui_config.n_cpu = values["n_cpu"]
            self.gui_config.f0method = ["pm", "harvest", "crepe", "rmvpe"][
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
                self.gui_config.pitch,
                self.gui_config.pth_path,
                self.gui_config.index_path,
                self.gui_config.index_rate,
                self.gui_config.n_cpu,
                inp_q,
                opt_q,
                self.config,
                self.rvc if hasattr(self, "rvc") else None,
            )
            self.gui_config.samplerate = self.rvc.tgt_sr
            self.zc = self.rvc.tgt_sr // 100
            self.block_frame = (
                int(
                    np.round(
                        self.gui_config.block_time
                        * self.gui_config.samplerate
                        / self.zc
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
            self.sola_search_frame = self.zc
            self.extra_frame = (
                int(
                    np.round(
                        self.gui_config.extra_time
                        * self.gui_config.samplerate
                        / self.zc
                    )
                )
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
            self.input_wav_res: torch.Tensor = torch.zeros(
                160 * self.input_wav.shape[0] // self.zc,
                device=self.config.device,
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
            self.sola_buffer: torch.Tensor = torch.zeros(
                self.crossfade_frame, device=self.config.device, dtype=torch.float32
            )
            self.nr_buffer: torch.Tensor = self.sola_buffer.clone()
            self.output_buffer: torch.Tensor = self.input_wav.clone()
            self.res_buffer: torch.Tensor = torch.zeros(
                2 * self.zc, device=self.config.device, dtype=torch.float32
            )
            self.valid_rate = 1 - (self.extra_frame - 1) / self.input_wav.shape[0]
            self.fade_in_window: torch.Tensor = (
                torch.sin(
                    0.5
                    * np.pi
                    * torch.linspace(
                        0.0,
                        1.0,
                        steps=self.crossfade_frame,
                        device=self.config.device,
                        dtype=torch.float32,
                    )
                )
                ** 2
            )
            self.fade_out_window: torch.Tensor = 1 - self.fade_in_window
            self.resampler = tat.Resample(
                orig_freq=self.gui_config.samplerate,
                new_freq=16000,
                dtype=torch.float32,
            ).to(self.config.device)
            self.tg = TorchGate(
                sr=self.gui_config.samplerate, n_fft=4 * self.zc, prop_decrease=0.9
            ).to(self.config.device)
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
                samplerate=self.gui_config.samplerate,
                dtype="float32",
            ) as stream:
                global stream_latency
                stream_latency = stream.latency[-1]
                while self.flag_vc:
                    time.sleep(self.gui_config.block_time)
                    printt("Audio block passed.")
            printt("ENDing VC")

        def audio_callback(
            self, indata: np.ndarray, outdata: np.ndarray, frames, times, status
        ):
            """
            音频处理
            """
            start_time = time.perf_counter()
            indata = librosa.to_mono(indata.T)
            if self.gui_config.threhold > -60:
                rms = librosa.feature.rms(
                    y=indata, frame_length=4 * self.zc, hop_length=self.zc
                )
                db_threhold = (
                    librosa.amplitude_to_db(rms, ref=1.0)[0] < self.gui_config.threhold
                )
                for i in range(db_threhold.shape[0]):
                    if db_threhold[i]:
                        indata[i * self.zc : (i + 1) * self.zc] = 0
            self.input_wav[: -self.block_frame] = self.input_wav[
                self.block_frame :
            ].clone()
            self.input_wav[-self.block_frame :] = torch.from_numpy(indata).to(
                self.config.device
            )
            self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[
                self.block_frame_16k :
            ].clone()
            # input noise reduction and resampling
            if self.gui_config.I_noise_reduce and self.function == "vc":
                input_wav = self.input_wav[
                    -self.crossfade_frame - self.block_frame - 2 * self.zc :
                ]
                input_wav = self.tg(
                    input_wav.unsqueeze(0), self.input_wav.unsqueeze(0)
                )[0, 2 * self.zc :]
                input_wav[: self.crossfade_frame] *= self.fade_in_window
                input_wav[: self.crossfade_frame] += (
                    self.nr_buffer * self.fade_out_window
                )
                self.nr_buffer[:] = input_wav[-self.crossfade_frame :]
                input_wav = torch.cat(
                    (self.res_buffer[:], input_wav[: self.block_frame])
                )
                self.res_buffer[:] = input_wav[-2 * self.zc :]
                self.input_wav_res[-self.block_frame_16k - 160 :] = self.resampler(
                    input_wav
                )[160:]
            else:
                self.input_wav_res[-self.block_frame_16k - 160 :] = self.resampler(
                    self.input_wav[-self.block_frame - 2 * self.zc :]
                )[160:]
            # infer
            if self.function == "vc":
                f0_extractor_frame = self.block_frame_16k + 800
                if self.gui_config.f0method == "rmvpe":
                    f0_extractor_frame = (
                        5120 * ((f0_extractor_frame - 1) // 5120 + 1) - 160
                    )
                infer_wav = self.rvc.infer(
                    self.input_wav_res,
                    self.input_wav_res[-f0_extractor_frame:].cpu().numpy(),
                    self.block_frame_16k,
                    self.valid_rate,
                    self.pitch,
                    self.pitchf,
                    self.gui_config.f0method,
                )
                infer_wav = infer_wav[
                    -self.crossfade_frame - self.sola_search_frame - self.block_frame :
                ]
            else:
                infer_wav = self.input_wav[
                    -self.crossfade_frame - self.sola_search_frame - self.block_frame :
                ].clone()
            # output noise reduction
            if (self.gui_config.O_noise_reduce and self.function == "vc") or (
                self.gui_config.I_noise_reduce and self.function == "im"
            ):
                self.output_buffer[: -self.block_frame] = self.output_buffer[
                    self.block_frame :
                ].clone()
                self.output_buffer[-self.block_frame :] = infer_wav[-self.block_frame :]
                infer_wav = self.tg(
                    infer_wav.unsqueeze(0), self.output_buffer.unsqueeze(0)
                ).squeeze(0)
            # volume envelop mixing
            if self.gui_config.rms_mix_rate < 1 and self.function == "vc":
                rms1 = librosa.feature.rms(
                    y=self.input_wav_res[-160 * infer_wav.shape[0] // self.zc :]
                    .cpu()
                    .numpy(),
                    frame_length=640,
                    hop_length=160,
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
                None, None, : self.crossfade_frame + self.sola_search_frame
            ]
            cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
            cor_den = torch.sqrt(
                F.conv1d(
                    conv_input**2,
                    torch.ones(1, 1, self.crossfade_frame, device=self.config.device),
                )
                + 1e-8
            )
            if sys.platform == "darwin":
                _, sola_offset = torch.max(cor_nom[0, 0] / cor_den[0, 0])
                sola_offset = sola_offset.item()
            else:
                sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
            printt("sola_offset = %d", int(sola_offset))
            infer_wav = infer_wav[
                sola_offset : sola_offset + self.block_frame + self.crossfade_frame
            ]
            infer_wav[: self.crossfade_frame] *= self.fade_in_window
            infer_wav[: self.crossfade_frame] += self.sola_buffer * self.fade_out_window
            self.sola_buffer[:] = infer_wav[-self.crossfade_frame :]
            if sys.platform == "darwin":
                outdata[:] = (
                    infer_wav[: -self.crossfade_frame].cpu().numpy()[:, np.newaxis]
                )
            else:
                outdata[:] = (
                    infer_wav[: -self.crossfade_frame].repeat(2, 1).t().cpu().numpy()
                )
            total_time = time.perf_counter() - start_time
            self.window["infer_time"].update(int(total_time * 1000))
            printt("Infer time: %.2f", total_time)

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
            printt("Input device: %s:%s", str(sd.default.device[0]), input_device)
            printt("Output device: %s:%s", str(sd.default.device[1]), output_device)

    gui = GUI()
