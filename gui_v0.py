import os, sys, traceback, re

import json

now_dir = os.getcwd()
sys.path.append(now_dir)
from config import Config

Config = Config()
import PySimpleGUI as sg
import sounddevice as sd
import noisereduce as nr
import numpy as np
from fairseq import checkpoint_utils
import librosa, torch, pyworld, faiss, time, threading
import torch.nn.functional as F
import torchaudio.transforms as tat
import scipy.signal as signal


# import matplotlib.pyplot as plt
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from i18n import I18nAuto

i18n = I18nAuto()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_dir = os.getcwd()


class RVC:
    def __init__(
        self, key, hubert_path, pth_path, index_path, npy_path, index_rate
    ) -> None:
        """
        初始化
        """
        try:
            self.f0_up_key = key
            self.time_step = 160 / 16000 * 1000
            self.f0_min = 50
            self.f0_max = 1100
            self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
            self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
            self.sr = 16000
            self.window = 160
            if index_rate != 0:
                self.index = faiss.read_index(index_path)
                # self.big_npy = np.load(npy_path)
                self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
                print("index search enabled")
            self.index_rate = index_rate
            model_path = hubert_path
            print("load model(s) from {}".format(model_path))
            models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
                [model_path],
                suffix="",
            )
            self.model = models[0]
            self.model = self.model.to(device)
            if Config.is_half:
                self.model = self.model.half()
            else:
                self.model = self.model.float()
            self.model.eval()
            cpt = torch.load(pth_path, map_location="cpu")
            self.tgt_sr = cpt["config"][-1]
            cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
            self.if_f0 = cpt.get("f0", 1)
            self.version = cpt.get("version", "v1")
            if self.version == "v1":
                if self.if_f0 == 1:
                    self.net_g = SynthesizerTrnMs256NSFsid(
                        *cpt["config"], is_half=Config.is_half
                    )
                else:
                    self.net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
            elif self.version == "v2":
                if self.if_f0 == 1:
                    self.net_g = SynthesizerTrnMs768NSFsid(
                        *cpt["config"], is_half=Config.is_half
                    )
                else:
                    self.net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
            del self.net_g.enc_q
            print(self.net_g.load_state_dict(cpt["weight"], strict=False))
            self.net_g.eval().to(device)
            if Config.is_half:
                self.net_g = self.net_g.half()
            else:
                self.net_g = self.net_g.float()
        except:
            print(traceback.format_exc())

    def get_f0(self, x, f0_up_key, inp_f0=None):
        x_pad = 1
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        f0, t = pyworld.harvest(
            x.astype(np.double),
            fs=self.sr,
            f0_ceil=f0_max,
            f0_floor=f0_min,
            frame_period=10,
        )
        f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.sr)
        f0 = signal.medfilt(f0, 3)
        f0 *= pow(2, f0_up_key / 12)
        # with open("test.txt","w")as f:f.write("\n".join([str(i)for i in f0.tolist()]))
        tf0 = self.sr // self.window  # 每秒f0点数
        if inp_f0 is not None:
            delta_t = np.round(
                (inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1
            ).astype("int16")
            replace_f0 = np.interp(
                list(range(delta_t)), inp_f0[:, 0] * 100, inp_f0[:, 1]
            )
            shape = f0[x_pad * tf0 : x_pad * tf0 + len(replace_f0)].shape[0]
            f0[x_pad * tf0 : x_pad * tf0 + len(replace_f0)] = replace_f0[:shape]
        # with open("test_opt.txt","w")as f:f.write("\n".join([str(i)for i in f0.tolist()]))
        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int32)
        return f0_coarse, f0bak  # 1-0

    def infer(self, feats: torch.Tensor) -> np.ndarray:
        """
        推理函数
        """
        audio = feats.clone().cpu().numpy()
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        if Config.is_half:
            feats = feats.half()
        else:
            feats = feats.float()
        inputs = {
            "source": feats.to(device),
            "padding_mask": padding_mask.to(device),
            "output_layer": 9 if self.version == "v1" else 12,
        }
        torch.cuda.synchronize()
        with torch.no_grad():
            logits = self.model.extract_features(**inputs)
            feats = (
                self.model.final_proj(logits[0]) if self.version == "v1" else logits[0]
            )

        ####索引优化
        try:
            if (
                hasattr(self, "index")
                and hasattr(self, "big_npy")
                and self.index_rate != 0
            ):
                npy = feats[0].cpu().numpy().astype("float32")
                score, ix = self.index.search(npy, k=8)
                weight = np.square(1 / score)
                weight /= weight.sum(axis=1, keepdims=True)
                npy = np.sum(self.big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
                if Config.is_half:
                    npy = npy.astype("float16")
                feats = (
                    torch.from_numpy(npy).unsqueeze(0).to(device) * self.index_rate
                    + (1 - self.index_rate) * feats
                )
            else:
                print("index search FAIL or disabled")
        except:
            traceback.print_exc()
            print("index search FAIL")
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        torch.cuda.synchronize()
        print(feats.shape)
        if self.if_f0 == 1:
            pitch, pitchf = self.get_f0(audio, self.f0_up_key)
            p_len = min(feats.shape[1], 13000, pitch.shape[0])  # 太大了爆显存
        else:
            pitch, pitchf = None, None
            p_len = min(feats.shape[1], 13000)  # 太大了爆显存
        torch.cuda.synchronize()
        # print(feats.shape,pitch.shape)
        feats = feats[:, :p_len, :]
        if self.if_f0 == 1:
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            pitch = torch.LongTensor(pitch).unsqueeze(0).to(device)
            pitchf = torch.FloatTensor(pitchf).unsqueeze(0).to(device)
        p_len = torch.LongTensor([p_len]).to(device)
        ii = 0  # sid
        sid = torch.LongTensor([ii]).to(device)
        with torch.no_grad():
            if self.if_f0 == 1:
                infered_audio = (
                    self.net_g.infer(feats, p_len, pitch, pitchf, sid)[0][0, 0]
                    .data.cpu()
                    .float()
                )
            else:
                infered_audio = (
                    self.net_g.infer(feats, p_len, sid)[0][0, 0].data.cpu().float()
                )
        torch.cuda.synchronize()
        return infered_audio


class GUIConfig:
    def __init__(self) -> None:
        self.hubert_path: str = ""
        self.pth_path: str = ""
        self.index_path: str = ""
        self.npy_path: str = ""
        self.pitch: int = 12
        self.samplerate: int = 44100
        self.block_time: float = 1.0  # s
        self.buffer_num: int = 1
        self.threhold: int = -30
        self.crossfade_time: float = 0.08
        self.extra_time: float = 0.04
        self.I_noise_reduce = False
        self.O_noise_reduce = False
        self.index_rate = 0.3


class GUI:
    def __init__(self) -> None:
        self.config = GUIConfig()
        self.flag_vc = False

        self.launcher()

    def load(self):
        (
            input_devices,
            output_devices,
            input_devices_indices,
            output_devices_indices,
        ) = self.get_devices()
        try:
            with open("values1.json", "r") as j:
                data = json.load(j)
        except:
            with open("values1.json", "w") as j:
                data = {
                    "pth_path": "",
                    "index_path": "",
                    "sg_input_device": input_devices[
                        input_devices_indices.index(sd.default.device[0])
                    ],
                    "sg_output_device": output_devices[
                        output_devices_indices.index(sd.default.device[1])
                    ],
                    "threhold": "-45",
                    "pitch": "0",
                    "index_rate": "0",
                    "block_time": "1",
                    "crossfade_length": "0.04",
                    "extra_time": "1",
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
                                default_text="hubert_base.pt",
                                key="hubert_path",
                                disabled=True,
                            ),
                            sg.FileBrowse(
                                i18n("Hubert模型"),
                                initial_folder=os.path.join(os.getcwd()),
                                file_types=(("pt files", "*.pt"),),
                            ),
                        ],
                        [
                            sg.Input(
                                default_text=data.get("pth_path", ""),
                                key="pth_path",
                            ),
                            sg.FileBrowse(
                                i18n("选择.pth文件"),
                                initial_folder=os.path.join(os.getcwd(), "weights"),
                                file_types=(("weight files", "*.pth"),),
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
                                file_types=(("index files", "*.index"),),
                            ),
                        ],
                        [
                            sg.Input(
                                default_text="你不需要填写这个You don't need write this.",
                                key="npy_path",
                                disabled=True,
                            ),
                            sg.FileBrowse(
                                i18n("选择.npy文件"),
                                initial_folder=os.path.join(os.getcwd(), "logs"),
                                file_types=(("feature files", "*.npy"),),
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
                                range=(0.1, 3.0),
                                key="block_time",
                                resolution=0.1,
                                orientation="h",
                                default_value=data.get("block_time", ""),
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
                            ),
                        ],
                        [
                            sg.Text(i18n("额外推理时长")),
                            sg.Slider(
                                range=(0.05, 3.00),
                                key="extra_time",
                                resolution=0.01,
                                orientation="h",
                                default_value=data.get("extra_time", ""),
                            ),
                        ],
                        [
                            sg.Checkbox(i18n("输入降噪"), key="I_noise_reduce"),
                            sg.Checkbox(i18n("输出降噪"), key="O_noise_reduce"),
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
        self.window = sg.Window("RVC - GUI", layout=layout)
        self.event_handler()

    def event_handler(self):
        while True:
            event, values = self.window.read()
            if event == sg.WINDOW_CLOSED:
                self.flag_vc = False
                exit()
            if event == "start_vc" and self.flag_vc == False:
                if self.set_values(values) == True:
                    print("using_cuda:" + str(torch.cuda.is_available()))
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
                    }
                    with open("values1.json", "w") as j:
                        json.dump(settings, j)
            if event == "stop_vc" and self.flag_vc == True:
                self.flag_vc = False

    def set_values(self, values):
        if len(values["pth_path"].strip()) == 0:
            sg.popup(i18n("请选择pth文件"))
            return False
        if len(values["index_path"].strip()) == 0:
            sg.popup(i18n("请选择index文件"))
            return False
        pattern = re.compile("[^\x00-\x7F]+")
        if pattern.findall(values["hubert_path"]):
            sg.popup(i18n("hubert模型路径不可包含中文"))
            return False
        if pattern.findall(values["pth_path"]):
            sg.popup(i18n("pth文件路径不可包含中文"))
            return False
        if pattern.findall(values["index_path"]):
            sg.popup(i18n("index文件路径不可包含中文"))
            return False
        self.set_devices(values["sg_input_device"], values["sg_output_device"])
        self.config.hubert_path = os.path.join(current_dir, "hubert_base.pt")
        self.config.pth_path = values["pth_path"]
        self.config.index_path = values["index_path"]
        self.config.npy_path = values["npy_path"]
        self.config.threhold = values["threhold"]
        self.config.pitch = values["pitch"]
        self.config.block_time = values["block_time"]
        self.config.crossfade_time = values["crossfade_length"]
        self.config.extra_time = values["extra_time"]
        self.config.I_noise_reduce = values["I_noise_reduce"]
        self.config.O_noise_reduce = values["O_noise_reduce"]
        self.config.index_rate = values["index_rate"]
        return True

    def start_vc(self):
        torch.cuda.empty_cache()
        self.flag_vc = True
        self.block_frame = int(self.config.block_time * self.config.samplerate)
        self.crossfade_frame = int(self.config.crossfade_time * self.config.samplerate)
        self.sola_search_frame = int(0.012 * self.config.samplerate)
        self.delay_frame = int(0.01 * self.config.samplerate)  # 往前预留0.02s
        self.extra_frame = int(self.config.extra_time * self.config.samplerate)
        self.rvc = None
        self.rvc = RVC(
            self.config.pitch,
            self.config.hubert_path,
            self.config.pth_path,
            self.config.index_path,
            self.config.npy_path,
            self.config.index_rate,
        )
        self.input_wav: np.ndarray = np.zeros(
            self.extra_frame
            + self.crossfade_frame
            + self.sola_search_frame
            + self.block_frame,
            dtype="float32",
        )
        self.output_wav: torch.Tensor = torch.zeros(
            self.block_frame, device=device, dtype=torch.float32
        )
        self.sola_buffer: torch.Tensor = torch.zeros(
            self.crossfade_frame, device=device, dtype=torch.float32
        )
        self.fade_in_window: torch.Tensor = torch.linspace(
            0.0, 1.0, steps=self.crossfade_frame, device=device, dtype=torch.float32
        )
        self.fade_out_window: torch.Tensor = 1 - self.fade_in_window
        self.resampler1 = tat.Resample(
            orig_freq=self.config.samplerate, new_freq=16000, dtype=torch.float32
        )
        self.resampler2 = tat.Resample(
            orig_freq=self.rvc.tgt_sr,
            new_freq=self.config.samplerate,
            dtype=torch.float32,
        )
        thread_vc = threading.Thread(target=self.soundinput)
        thread_vc.start()

    def soundinput(self):
        """
        接受音频输入
        """
        with sd.Stream(
            channels=2,
            callback=self.audio_callback,
            blocksize=self.block_frame,
            samplerate=self.config.samplerate,
            dtype="float32",
        ):
            while self.flag_vc:
                time.sleep(self.config.block_time)
                print("Audio block passed.")
        print("ENDing VC")

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
        db_threhold = librosa.amplitude_to_db(rms, ref=1.0)[0] < self.config.threhold
        # print(rms.shape,db.shape,db)
        for i in range(db_threhold.shape[0]):
            if db_threhold[i]:
                indata[i * hop_length : (i + 1) * hop_length] = 0
        self.input_wav[:] = np.append(self.input_wav[self.block_frame :], indata)

        # infer
        print("input_wav:" + str(self.input_wav.shape))
        # print('infered_wav:'+str(infer_wav.shape))
        infer_wav: torch.Tensor = self.resampler2(
            self.rvc.infer(self.resampler1(torch.from_numpy(self.input_wav)))
        )[-self.crossfade_frame - self.sola_search_frame - self.block_frame :].to(
            device
        )
        print("infer_wav:" + str(infer_wav.shape))

        # SOLA algorithm from https://github.com/yxlllc/DDSP-SVC
        cor_nom = F.conv1d(
            infer_wav[None, None, : self.crossfade_frame + self.sola_search_frame],
            self.sola_buffer[None, None, :],
        )
        cor_den = torch.sqrt(
            F.conv1d(
                infer_wav[None, None, : self.crossfade_frame + self.sola_search_frame]
                ** 2,
                torch.ones(1, 1, self.crossfade_frame, device=device),
            )
            + 1e-8
        )
        sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
        print("sola offset: " + str(int(sola_offset)))

        # crossfade
        self.output_wav[:] = infer_wav[sola_offset : sola_offset + self.block_frame]
        self.output_wav[: self.crossfade_frame] *= self.fade_in_window
        self.output_wav[: self.crossfade_frame] += self.sola_buffer[:]
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
            outdata[:] = np.tile(
                nr.reduce_noise(
                    y=self.output_wav[:].cpu().numpy(), sr=self.config.samplerate
                ),
                (2, 1),
            ).T
        else:
            outdata[:] = self.output_wav[:].repeat(2, 1).t().cpu().numpy()
        total_time = time.perf_counter() - start_time
        self.window["infer_time"].update(int(total_time * 1000))
        print("infer time:" + str(total_time))

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
        sd.default.device[0] = input_device_indices[input_devices.index(input_device)]
        sd.default.device[1] = output_device_indices[
            output_devices.index(output_device)
        ]
        print("input device:" + str(sd.default.device[0]) + ":" + str(input_device))
        print("output device:" + str(sd.default.device[1]) + ":" + str(output_device))


gui = GUI()
