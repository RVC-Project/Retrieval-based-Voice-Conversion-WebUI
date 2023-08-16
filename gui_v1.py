from collections import OrderedDict
import os, sys,pdb
os.environ["OMP_NUM_THREADS"]="2"
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)
import multiprocessing


class Harvest(multiprocessing.Process):
    def __init__(self, inp_q, opt_q):
        multiprocessing.Process.__init__(self)
        self.inp_q = inp_q
        self.opt_q = opt_q

    def run(self):
        import numpy as np, pyworld

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
    from multiprocessing import Queue
    from queue import Empty
    import numpy as np
    import multiprocessing
    import traceback, re
    import json
    import PySimpleGUI as sg
    import sounddevice as sd
    import noisereduce as nr
    from multiprocessing import cpu_count
    import librosa, torch, time, threading
    import torch.nn.functional as F
    import torchaudio.transforms as tat
    from i18n import I18nAuto
    import rvc_for_realtime
    i18n = I18nAuto()
    device=rvc_for_realtime.config.device
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
            self.ad_features = False
            self.window_size = None
            self.num_models= 2

            self.launcher()

        def load(self):
            input_devices, output_devices, _, _ = self.get_devices()
            try:
                with open("values1.json", "r") as j:
                    data = json.load(j)
                    data["pm"] = data["f0method"] == "pm"
                    data["harvest"] = data["f0method"] == "harvest"
                    data["crepe"] = data["f0method"] == "crepe"
                    data["rmvpe"] = data["f0method"] == "rmvpe"
            except:
                with open("values1.json", "w") as j:
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
            self.layout = [
                [sg.Frame(
                            title="基础功能",
                            key="basic_features",
                            layout=
                            [[
                                sg.Frame(
                                    title=i18n("加载模型"),
                                    key="load_model",
                                    layout=[
                                        [
                                            sg.Input(
                                                default_text=data.get("pth_path", ""),
                                                key="pth_path",
                                            ),
                                            sg.FileBrowse(
                                                i18n("选择.pth文件"),
                                                initial_folder=os.path.join(os.getcwd(), "weights"),
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
                                        [
                                            sg.Text(i18n("音高算法")),
                                            sg.Radio(
                                                "pm",
                                                "f0method",
                                                key="pm",
                                                default=data.get("pm", "") == True,
                                            ),
                                            sg.Radio(
                                                "harvest",
                                                "f0method",
                                                key="harvest",
                                                default=data.get("harvest", "") == True,
                                            ),
                                            sg.Radio(
                                                "crepe",
                                                "f0method",
                                                key="crepe",
                                                default=data.get("crepe", "") == True,
                                            ),
                                            sg.Radio(
                                                "rmvpe",
                                                "f0method",
                                                key="rmvpe",
                                                default=data.get("rmvpe", "") == True,
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
                                                range=(0.09, 2.4),
                                                key="block_time",
                                                resolution=0.03,
                                                orientation="h",
                                                default_value=data.get("block_time", ""),
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
                                                range=(0.05, 5.00),
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
                            ],
                        ),
                        sg.Frame(
                            title=i18n("模型融合"),
                            key="ModelMix",
                            layout=[
                                    [       
                                        sg.Checkbox(i18n("开启"), key="ad_features",enable_events=True),
                                    ],
                                    [
                                        sg.Text(i18n("使用模型融合功能时，模型版本、模型采样率必须一致！")),
                                    ],
                                    [
                                        sg.Frame(title=i18n("模型路径"),
                                                    layout=[
                                                        [
                                                            sg.Frame(title=i18n("模型A"),
                                                                layout=[
                                                                        [
                                                                            # sg.Text("模型A"),
                                                                            sg.Input(
                                                                                default_text=data.get("pth_path_a",""),
                                                                                key="pth_path_a",
                                                                            ),
                                                                            sg.FileBrowse(
                                                                                i18n("选择.pth文件"),
                                                                                initial_folder=os.path.join(os.getcwd(), "weights"),
                                                                                file_types=((". pth"),),
                                                                            ),
                                                                        ],
                                                                ],
                                                            ),
                                                            
                                                        ],
                                                        [
                                                            sg.Frame(title=i18n("模型B"),
                                                                layout=[
                                                                        [
                                                                            # sg.Text("模型A"),
                                                                            sg.Input(
                                                                                default_text=data.get("pth_path_b",""),
                                                                                key="pth_path_b",
                                                                            ),
                                                                            sg.FileBrowse(
                                                                                i18n("选择.pth文件"),
                                                                                initial_folder=os.path.join(os.getcwd(), "weights"),
                                                                                file_types=((". pth"),),
                                                                            ),
                                                                        ],
                                                                ],
                                                            ),
                                                        ],
                                                        [
                                                            sg.Frame(title=i18n("模型C"),
                                                                     key="model_c",
                                                                     visible=False,
                                                                layout=[
                                                                        [
                                                                            # sg.Text("模型A"),
                                                                            sg.Input(
                                                                                default_text=data.get("pth_path_c",""),
                                                                                key="pth_path_c",
                                                                            ),
                                                                            sg.FileBrowse(
                                                                                i18n("选择.pth文件"),
                                                                                initial_folder=os.path.join(os.getcwd(), "weights"),
                                                                                file_types=((". pth"),),
                                                                            ),
                                                                        ],
                                                                ],
                                                            ),
                                                            
                                                        ],
                                                        [
                                                            sg.Frame(title=i18n("模型D"),
                                                                     key="model_d",
                                                                     visible=False,
                                                                layout=[
                                                                        [
                                                                            # sg.Text("模型A"),
                                                                            sg.Input(
                                                                                default_text=data.get("pth_path_d",""),
                                                                                key="pth_path_d",
                                                                            ),
                                                                            sg.FileBrowse(
                                                                                i18n("选择.pth文件"),
                                                                                initial_folder=os.path.join(os.getcwd(), "weights"),
                                                                                file_types=((". pth"),),
                                                                            ),
                                                                        ],
                                                                ],
                                                            ),
                                                        ],
                                                        [
                                                            sg.Frame(title="Index",
                                                                layout=[
                                                                        [
                                                                            # sg.Text("模型A"),
                                                                            sg.Input(
                                                                                default_text=data.get("path_index",""),
                                                                                key="path_index",
                                                                            ),
                                                                            sg.FileBrowse(
                                                                                i18n("选择.index文件"),
                                                                                initial_folder=os.path.join(os.getcwd(), "logs"),
                                                                                file_types=((". index"),),
                                                                            ),
                                                                        ],
                                                                ],
                                                            ),
                                                        ],
                                                        
                                                    ]
                                        )
                                    ],
                                    [
                                        sg.Frame(title=i18n("权重调节"),
                                                    layout=[
                                                        [
                                                            sg.Frame(title=i18n("模型A权重"),
                                                                layout=[
                                                                        [
                                                                            # sg.Text("模型A权重"),
                                                                            sg.Slider(
                                                                                range=(0, 1),
                                                                                key="weight_a",
                                                                                resolution=0.0001,
                                                                                orientation="h",
                                                                                default_value=data.get("weight_a",0.5),
                                                                                enable_events=True
                                                                            ),
                                                                            sg.Checkbox(i18n("锁定"), key="lock_a"),
                                                                        ],
                                                                    ]
                                                            ),
                                                            sg.Frame(title=i18n("模型B权重"),
                                                                layout=[
                                                                        [
                                                                            sg.Slider(
                                                                                range=(0, 1),
                                                                                key="weight_b",
                                                                                resolution=0.0001,
                                                                                orientation="h",
                                                                                default_value=data.get("weight_b",0.5),
                                                                                enable_events=True
                                                                            ),
                                                                            sg.Checkbox(i18n("锁定"), key="lock_b"),
                                                                        ],
                                                                    ]
                                                            ),
                                                        ],
                                                        [
                                                            sg.Frame(title=i18n("模型C权重"),
                                                                     key="weight_c_frame",
                                                                     visible=False,
                                                                layout=[
                                                                        [
                                                                            sg.Slider(
                                                                                range=(0, 1),
                                                                                key="weight_c",
                                                                                resolution=0.0001,
                                                                                orientation="h",
                                                                                default_value=data.get("weight_c",0),
                                                                                enable_events=True
                                                                            ),
                                                                            sg.Checkbox(i18n("锁定"), key="lock_c"),
                                                                        ],
                                                                    ]
                                                            ),
                                                            sg.Frame(title=i18n("模型D权重"),
                                                                     key="weight_d_frame",
                                                                     visible=False,
                                                                layout=[
                                                                        [
                                                                            sg.Slider(
                                                                                range=(0, 1),
                                                                                key="weight_d",
                                                                                resolution=0.0001,
                                                                                orientation="h",
                                                                                default_value=data.get("weight_d",0),
                                                                                enable_events=True
                                                                            ),
                                                                            sg.Checkbox(i18n("锁定"), key="lock_d"),
                                                                        ],
                                                                    ]
                                                            ),
                                                        ],
                                                    
                                            ])
                                    ],
                                    [
                                        sg.Button(i18n("增加模型"), key="add_model"),
                                        sg.Button(i18n("删除最后一个模型"), key="delete_model"),
                                    ],
                                    [
                                        sg.Text(i18n("最多4个模型, 最少2个模型, 权重和必须为1")),
                                    ]
   
                            ])
                    ],
                    [
                        sg.Button(i18n("开始音频转换"), key="start_vc"),
                        sg.Button(i18n("停止音频转换"), key="stop_vc"),
                        sg.Text(i18n("推理时间(ms):")),
                        sg.Text("0", key="infer_time"),
                    ],

            ]
            self.window = sg.Window("RVC - GUI", layout=self.layout)
            
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
                        print("using_cuda:" + str(torch.cuda.is_available()))
                        self.start_vc(values)
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
                            "pth_path_a":values["pth_path_a"],
                            "pth_path_b":values["pth_path_b"],
                            "pth_path_c":values["pth_path_c"],
                            "pth_path_d":values["pth_path_d"],
                            "path_index":values["path_index"],
                            "weight_a":values["weight_a"],
                            "weight_b":values["weight_b"],
                            "weight_c":values["weight_c"],
                            "weight_d":values["weight_d"],
                        }
                        with open("values1.json", "w") as j:
                            json.dump(settings, j)
                if event == "stop_vc" and self.flag_vc == True:
                    self.flag_vc = False

                if event == "ad_features":
                    self.ad_features = not self.ad_features
                    if self.ad_features:
                        self.window["load_model"].update(visible=not self.ad_features)
                        print("ModelMix已开启！")
                    else:
                        self.window["load_model"].update(visible=not self.ad_features)
                        print("ModelMix已关闭！")
                        
                if event == "add_model":
                    for i in ["c","d"]:
                        if not self.window[f"weight_{i}_frame"].visible:
                            self.window[f"weight_{i}_frame"].update(visible=True)
                            self.window[f"model_{i}"].update(visible=True)
                            self.num_models+=1
                            break
                    self.set_mix_weight("a",values)
                        
                if event == "delete_model":
                    n=["d","c"]
                    for i in n:
                        if self.window[f"weight_{i}_frame"].visible:
                            self.window[f"weight_{i}_frame"].update(visible=False)
                            self.window[f"model_{i}"].update(visible=False)
                            self.window[f"pth_path_{i}"].update("")
                            self.num_models-=1
                            break
                    self.set_mix_weight("a",values)

                if event == "weight_a" and self.flag_vc == False:
                    self.set_mix_weight("a",values)
                elif event == "weight_b" and self.flag_vc == False:
                    self.set_mix_weight("b",values)
                elif event == "weight_c" and self.flag_vc == False:
                    self.set_mix_weight("c",values)
                elif event == "weight_d" and self.flag_vc == False:
                    self.set_mix_weight("d",values)

        def set_mix_weight(self,x,values):
            w = 1
            n = ["a","b","c","d"][:self.num_models]
            n.remove(x)
            w_unlock=0.0
            w_lock=0
            _n = n.copy()
            for i in n:
                if values[f"lock_{i}"]:
                    w -= float(values[f"weight_{i}"])
                    w_lock += float(values[f"weight_{i}"])
                    _n.remove(i)
                else:
                    w_unlock += float(values[f"weight_{i}"])
            n=_n
            if float(values[f"weight_{x}"])>1-w_lock:
                self.window[f"weight_{x}"].update(str(1-w_lock))
                return 
            if len(n)==0 and not values[f"lock_{x}"]:
                self.window[f"weight_{x}"].update(str(1-w_lock))
                return
            
            w_unlock=1-w_lock-float(values[f"weight_{x}"])
            
            if n==[] and values[f"lock_{x}"]:
                self.window[f"weight_{x}"].update(str(1-w_lock))
            else:
                for i in n:
                    self.window[f"weight_{i}"].update(str(w_unlock/len(n)))
                    
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
        
        def mix_model(self,values):
            print("正在融合模型...")
            path_a = values["pth_path_a"]
            path_b = values["pth_path_b"]
            path_c = values["pth_path_c"]
            path_d = values["pth_path_d"]
            w_a = float(values[f"weight_a"])
            w_b = float(values[f"weight_b"])
            w_c = float(values[f"weight_c"]) 
            w_d = float(values[f"weight_d"])


            try:

                def extract(ckpt):
                    a = ckpt["model"]
                    opt = OrderedDict()
                    opt["weight"] = {}
                    for key in a.keys():
                        if "enc_q" in key:
                            continue
                        opt["weight"][key] = a[key]
                    return opt

                ckpt_a = torch.load(path_a, map_location="cpu")
                ckpt_b = torch.load(path_b, map_location="cpu")
                ckpt_c = torch.load(path_c, map_location="cpu") if path_c!="" else None
                ckpt_d = torch.load(path_d, map_location="cpu") if path_d!="" else None
                
                def f1(m):
                    if "model" in m:
                        m = extract(m)
                    else:
                        m = m["weight"]
                    return m
                cfg={}
                cfg["config"] = ckpt_a["config"]
                cfg["sr"] = ckpt_a["sr"]
                cfg["f0"] = ckpt_a["f0"]
                cfg["version"] = ckpt_a["version"]
                cfg["info"] = ckpt_a["info"]
                ckpt_a = f1(ckpt_a)
                ckpt_b = f1(ckpt_b)
                ckpt_c = f1(ckpt_c) if ckpt_c else None
                ckpt_d = f1(ckpt_d) if ckpt_d else None

                same_flag = True
                if sorted(list(ckpt_a.keys())) != sorted(list(ckpt_b.keys())):
                    same_flag = False
                if ckpt_c:
                    if sorted(list(ckpt_a.keys())) != sorted(list(ckpt_c.keys())):
                        same_flag = False
                if ckpt_d:
                    if sorted(list(ckpt_a.keys())) != sorted(list(ckpt_d.keys())):
                        same_flag = False
                        
                if not same_flag:
                    raise ValueError("模型架构不同，无法融合！(Fail to merge the models. The model architectures are not the same.)")
                
                def merge(m1,m2,m3=None,m4=None,alpha1=0.5,alpha2=0.5,alpha3=0,alpha4=0):
                    opt = OrderedDict()
                    opt["weight"] = {}
                    for key in m1.keys():
                        # try:
                        if key == "emb_g.weight" and m1[key].shape != m2[key].shape:
                            min_shape0 = min(m1[key].shape[0], m2[key].shape[0],m3[key].shape[0], m4[key].shape[0])
                            opt["weight"][key] = alpha1 * (m1[key][:min_shape0].float()) + alpha2 * (m2[key][:min_shape0].float())
                            if m3:
                                min_shape0=min(min_shape0,m3[key].shape)
                                opt["weight"][key] += alpha3 * (m3[key][:min_shape0].float())
                            if m4:
                                min_shape0=min(min_shape0,m4[key].shape)
                                opt["weight"][key] += alpha4 * (m4[key][:min_shape0].float())
                            opt["weight"][key]=opt["weight"][key].half()
                        else:
                            opt["weight"][key] = alpha1 * (m1[key].float()) + alpha2 * (m2[key].float())
                            if m3:
                                opt["weight"][key] = opt["weight"][key]+alpha3 * (m3[key].float())
                            if m4:
                                opt["weight"][key] = opt["weight"][key]+alpha4* (m4[key].float())
                            opt["weight"][key]=opt["weight"][key].half()
                    return opt
                
                model_list=[path_a,path_b]
                model_list.append(path_c) if path_c!="" else None
                model_list.append(path_b) if path_b!="" else None
                alpha_list = [w_a,w_b]
                alpha_list.append(w_c) if path_c!="" else None
                alpha_list.append(w_d) if path_d!="" else None
                print(f"正在融合下列模型：\n{model_list}\n权重：\n{alpha_list}")
                opt=merge(ckpt_a,ckpt_b,ckpt_c,ckpt_d,w_a,w_b,w_c,w_d)
                    # except:
                #     pdb.set_trace()
                opt["config"] = cfg["config"]
                """
                if(sr=="40k"):opt["config"] = [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10, 10, 2, 2], 512, [16, 16, 4, 4,4], 109, 256, 40000]
                elif(sr=="48k"):opt["config"] = [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10,6,2,2,2], 512, [16, 16, 4, 4], 109, 256, 48000]
                elif(sr=="32k"):opt["config"] = [513, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10, 4, 2, 2, 2], 512, [16, 16, 4, 4,4], 109, 256, 32000]
                """
                opt["sr"] = cfg["sr"]
                opt["f0"] = cfg["f0"]
                opt["version"] = cfg["version"]
                opt["info"] = cfg["info"]

                save_path = "weights/mixed.pth"
                torch.save(opt, save_path)
                print("融合完成！")
                return save_path
                # return "Success."
            except:
                return traceback.format_exc()

        def start_vc(self,values=None):
            torch.cuda.empty_cache()
            self.flag_vc = True
            if self.ad_features:
                self.config.pth_path=self.mix_model(values)
                index_path=values["path_index"]
                self.config.index_path = index_path if index_path!="" else self.config.index_path
            print(f"loading model from {self.config.pth_path}")
            if self.config.index_rate==0:
                print(f"index disabled!")
            else:
                print(f"index enabled. the file from {self.config.index_path}")
            self.rvc = rvc_for_realtime.RVC(
                self.config.pitch,
                self.config.pth_path,
                self.config.index_path,
                self.config.index_rate,
                self.config.n_cpu,
                inp_q,
                opt_q,
                device,
            )
            self.config.samplerate = self.rvc.tgt_sr
            self.config.crossfade_time = min(
                self.config.crossfade_time, self.config.block_time
            )
            self.block_frame = int(self.config.block_time * self.config.samplerate)
            self.crossfade_frame = int(
                self.config.crossfade_time * self.config.samplerate
            )
            self.sola_search_frame = int(0.01 * self.config.samplerate)
            self.extra_frame = int(self.config.extra_time * self.config.samplerate)
            self.zc = self.rvc.tgt_sr // 100
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
            self.fade_in_window: torch.Tensor = torch.linspace(
                0.0, 1.0, steps=self.crossfade_frame, device=device, dtype=torch.float32
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
            if self.config.threhold > -60:
                db_threhold = (
                    librosa.amplitude_to_db(rms, ref=1.0)[0] < self.config.threhold
                )
                for i in range(db_threhold.shape[0]):
                    if db_threhold[i]:
                        indata[i * hop_length : (i + 1) * hop_length] = 0
            self.input_wav[:] = np.append(self.input_wav[self.block_frame :], indata)
            # infer
            inp = torch.from_numpy(self.input_wav).to(device)
            res1 = self.resampler(inp)
            ###55%
            rate1 = self.block_frame / (
                self.extra_frame
                + self.crossfade_frame
                + self.sola_search_frame
                + self.block_frame
            )
            rate2 = (
                self.crossfade_frame + self.sola_search_frame + self.block_frame
            ) / (
                self.extra_frame
                + self.crossfade_frame
                + self.sola_search_frame
                + self.block_frame
            )
            res2 = self.rvc.infer(
                res1,
                res1[-self.block_frame :].cpu().numpy(),
                rate1,
                rate2,
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
            print("sola offset: " + str(int(sola_offset)))
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
            sd.default.device[0] = input_device_indices[
                input_devices.index(input_device)
            ]
            sd.default.device[1] = output_device_indices[
                output_devices.index(output_device)
            ]
            print("input device:" + str(sd.default.device[0]) + ":" + str(input_device))
            print(
                "output device:" + str(sd.default.device[1]) + ":" + str(output_device)
            )

    gui = GUI()
