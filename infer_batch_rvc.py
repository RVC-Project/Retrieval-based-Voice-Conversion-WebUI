"""
v1
runtime\python.exe myinfer-v2-0528.py 0 "E:\codes\py39\RVC-beta\todo-songs" "E:\codes\py39\logs\mi-test\added_IVF677_Flat_nprobe_7.index" harvest "E:\codes\py39\RVC-beta\output" "E:\codes\py39\test-20230416b\weights\mi-test.pth" 0.66 cuda:0 True 3 0 1 0.33
v2
runtime\python.exe myinfer-v2-0528.py 0 "E:\codes\py39\RVC-beta\todo-songs" "E:\codes\py39\test-20230416b\logs\mi-test-v2\aadded_IVF677_Flat_nprobe_1_v2.index" harvest "E:\codes\py39\RVC-beta\output_v2" "E:\codes\py39\test-20230416b\weights\mi-test-v2.pth" 0.66 cuda:0 True 3 0 1 0.33
"""
import os, sys, pdb, torch

now_dir = os.getcwd()
sys.path.append(now_dir)
import sys
import torch
import tqdm as tq
from multiprocessing import cpu_count


class Config:
    def __init__(self, device, is_half):
        self.device = device
        self.is_half = is_half
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                print("16系/10系显卡和P40强制单精度")
                self.is_half = False
                for config_file in ["32k.json", "40k.json", "48k.json"]:
                    with open(f"configs/{config_file}", "r") as f:
                        strr = f.read().replace("true", "false")
                    with open(f"configs/{config_file}", "w") as f:
                        f.write(strr)
                with open("trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
            else:
                self.gpu_name = None
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            if self.gpu_mem <= 4:
                with open("trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
        elif torch.backends.mps.is_available():
            print("没有发现支持的N卡, 使用MPS进行推理")
            self.device = "mps"
        else:
            print("没有发现支持的N卡, 使用CPU进行推理")
            self.device = "cpu"
            self.is_half = True

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            # 6G显存配置
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G显存配置
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem != None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max


f0up_key = sys.argv[1]
input_path = sys.argv[2]
index_path = sys.argv[3]
f0method = sys.argv[4]  # harvest or pm
opt_path = sys.argv[5]
model_path = sys.argv[6]
index_rate = float(sys.argv[7])
device = sys.argv[8]
is_half = sys.argv[9].lower() != "false"
filter_radius = int(sys.argv[10])
resample_sr = int(sys.argv[11])
rms_mix_rate = float(sys.argv[12])
protect = float(sys.argv[13])
print(sys.argv)
config = Config(device, is_half)
now_dir = os.getcwd()
sys.path.append(now_dir)
from vc_infer_pipeline import VC
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from lib.audio import load_audio
from fairseq import checkpoint_utils
from scipy.io import wavfile

hubert_model = None


def load_hubert():
    global hubert_model
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(device)
    if is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()


def vc_single(sid, input_audio, f0_up_key, f0_file, f0_method, file_index, index_rate):
    global tgt_sr, net_g, vc, hubert_model, version
    if input_audio is None:
        return "You need to upload an audio", None
    f0_up_key = int(f0_up_key)
    audio = load_audio(input_audio, 16000)
    times = [0, 0, 0]
    if hubert_model == None:
        load_hubert()
    if_f0 = cpt.get("f0", 1)
    # audio_opt=vc.pipeline(hubert_model,net_g,sid,audio,times,f0_up_key,f0_method,file_index,file_big_npy,index_rate,if_f0,f0_file=f0_file)
    audio_opt = vc.pipeline(
        hubert_model,
        net_g,
        sid,
        audio,
        input_audio,
        times,
        f0_up_key,
        f0_method,
        file_index,
        index_rate,
        if_f0,
        filter_radius,
        tgt_sr,
        resample_sr,
        rms_mix_rate,
        version,
        protect,
        f0_file=f0_file,
    )
    print(times)
    return audio_opt


def get_vc(model_path):
    global n_spk, tgt_sr, net_g, vc, cpt, device, is_half, version
    print("loading pth %s" % model_path)
    cpt = torch.load(model_path, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:  #
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))  # 不加这一行清不干净，真奇葩
    net_g.eval().to(device)
    if is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]
    # return {"visible": True,"maximum": n_spk, "__type__": "update"}


get_vc(model_path)
audios = os.listdir(input_path)
for file in tq.tqdm(audios):
    if file.endswith(".wav"):
        file_path = input_path + "/" + file
        wav_opt = vc_single(
            0, file_path, f0up_key, None, f0method, index_path, index_rate
        )
        out_path = opt_path + "/" + file
        wavfile.write(out_path, tgt_sr, wav_opt)
