import os, sys, pdb, torch
import argparse
import glob
import sys
import torch
from multiprocessing import cpu_count
import argparse
import time

start_time = time.time()

now_dir = os.getcwd()
sys.path.append(now_dir)

parser = argparse.ArgumentParser()

# yapf: disable

parser.add_argument('input_audio', help='输入的音频文件')
parser.add_argument('-m', '--model', type=str, help='\weights路径下的pth文件')
parser.add_argument('-i', '--index', type=str, help='索引文件路径')
parser.add_argument('-f', '--f0method', type=str, help='f0的方法，可选‘pm’ ‘harvest’', default='harvest')
parser.add_argument('-k', '--key', type=int, help='音调改变', default=0)
parser.add_argument('-ir', '--index_rate', type=float, help='索引比例（取值0~1，越趋近于1理论上音色泄露更少）', default=0.7)
parser.add_argument('-d', '--device', type=str, help='使用的设备，默认 cuda:0 ', default='cuda:0')
parser.add_argument('-fp', '--is_half', action='store_true', help='是否使用半精度运算')
parser.add_argument('-o', '--output',type=str, default='')
parser.add_argument('-fr', '--filter_radius', type=int, help='对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音', default=3)
parser.add_argument('-s', '--tgt_sr', type=int, help='目标采样率', default=44100)
parser.add_argument('-rs', '--resample_sr',type=int, help='后处理重采样至最终采样率，0为不进行重采样', default=0 )
parser.add_argument('-rms', '--rms_mix_rate', type=float, help='输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络', default=1)
parser.add_argument('-v', '--version', type=str, help='模型版本', default='v1')
args = parser.parse_args()

# yapf: enable

class Config:

    def __init__(self, device, is_half):
        self.device = device
        self.is_half = is_half
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config(
        )

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                    or "P40" in self.gpu_name.upper()
                    or "1060" in self.gpu_name or "1070" in self.gpu_name
                    or "1080" in self.gpu_name):
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
                torch.cuda.get_device_properties(i_device).total_memory /
                1024 / 1024 / 1024 + 0.4)
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


# f0up_key=sys.argv[1]
# input_path=sys.argv[2]
# index_path=sys.argv[3]
# f0method=sys.argv[4]#harvest or pm
# opt_path=sys.argv[5]
# model_path=sys.argv[6]
# index_rate=float(sys.argv[7])
# device=sys.argv[8]
# is_half=bool(sys.argv[9])
# print(sys.argv)
config = Config(args.device, args.is_half)
now_dir = os.getcwd()
sys.path.append(now_dir)
from vc_infer_pipeline import VC
from infer_pack.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono
from my_utils import load_audio
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
    hubert_model = hubert_model.to(args.device)
    if (args.is_half): hubert_model = hubert_model.half()
    else: hubert_model = hubert_model.float()
    hubert_model.eval()


def vc_single(sid, input_audio, f0_up_key, f0_file, f0_method, file_index,
              index_rate, filter_radius, resample_sr, rms_mix_rate, version):
    global tgt_sr, net_g, vc, hubert_model
    if input_audio is None: return "You need to upload an audio", None
    f0_up_key = int(f0_up_key)
    audio = load_audio(input_audio, 16000)
    times = [0, 0, 0]
    if (hubert_model == None): load_hubert()
    if_f0 = cpt.get("f0", 1)
    # audio_opt=vc.pipeline(hubert_model,net_g,sid,audio,times,f0_up_key,f0_method,file_index,file_big_npy,index_rate,if_f0,f0_file=f0_file)
    audio_opt = vc.pipeline(hubert_model,
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
                            f0_file=f0_file)
    print(times)
    return audio_opt


def get_vc(model_path):
    global n_spk, tgt_sr, net_g, vc, cpt, args
    print("loading pth %s" % model_path)
    cpt = torch.load(model_path, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  #n_spk
    if_f0 = cpt.get("f0", 1)
    if (if_f0 == 1):
        net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=args.is_half)
    else:
        net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))  # 不加这一行清不干净，真奇葩
    net_g.eval().to(args.device)
    if (args.is_half): net_g = net_g.half()
    else: net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]
    # return {"visible": True,"maximum": n_spk, "__type__": "update"}


get_vc(args.model)
wav_opt = vc_single(0, args.input_audio, args.key, None, args.f0method,
                    args.index, args.index_rate, args.filter_radius,
                    args.resample_sr, args.rms_mix_rate, args.version)


t = time.strftime('%Y%m%d%H%M%S', time.localtime())
if not args.output:
    input_name = os.path.basename(args.input_audio).split('.')[0]
    model_name = os.path.basename(args.model).split('.')[0]
    opt_path = f'output/{input_name}_{model_name}_{args.index_rate}_{args.key}_{args.f0method}_{t}.wav'
else:
    opt_path = args.output
wavfile.write(opt_path, tgt_sr, wav_opt)

end_time = time.time()
used_time = end_time - start_time

print(f'done, used {used_time} seconds.')