"""

对源特征进行检索
"""
import os
import logging

logger = logging.getLogger(__name__)

import parselmouth
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# import torchcrepe
from time import time as ttime

# import pyworld
import librosa
import numpy as np
import soundfile as sf
import torch.nn.functional as F
from fairseq import checkpoint_utils

# from models import SynthesizerTrn256#hifigan_nonsf
# from lib.infer_pack.models import SynthesizerTrn256NSF as SynthesizerTrn256#hifigan_nsf
from infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid as SynthesizerTrn256,
)  # hifigan_nsf
from scipy.io import wavfile

# from lib.infer_pack.models import SynthesizerTrnMs256NSFsid_sim as SynthesizerTrn256#hifigan_nsf
# from models import SynthesizerTrn256NSFsim as SynthesizerTrn256#hifigan_nsf
# from models import SynthesizerTrn256NSFsimFlow as SynthesizerTrn256#hifigan_nsf


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r"E:\codes\py39\vits_vc_gpu_train\assets\hubert\hubert_base.pt"  #
logger.info("Load model(s) from {}".format(model_path))
models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
    [model_path],
    suffix="",
)
model = models[0]
model = model.to(device)
model = model.half()
model.eval()

# net_g = SynthesizerTrn256(1025,32,192,192,768,2,6,3,0.1,"1", [3,7,11],[[1,3,5], [1,3,5], [1,3,5]],[10,10,2,2],512,[16,16,4,4],183,256,is_half=True)#hifigan#512#256
# net_g = SynthesizerTrn256(1025,32,192,192,768,2,6,3,0.1,"1", [3,7,11],[[1,3,5], [1,3,5], [1,3,5]],[10,10,2,2],512,[16,16,4,4],109,256,is_half=True)#hifigan#512#256
net_g = SynthesizerTrn256(
    1025,
    32,
    192,
    192,
    768,
    2,
    6,
    3,
    0,
    "1",
    [3, 7, 11],
    [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    [10, 10, 2, 2],
    512,
    [16, 16, 4, 4],
    183,
    256,
    is_half=True,
)  # hifigan#512#256#no_dropout
# net_g = SynthesizerTrn256(1025,32,192,192,768,2,3,3,0.1,"1", [3,7,11],[[1,3,5], [1,3,5], [1,3,5]],[10,10,2,2],512,[16,16,4,4],0)#ts3
# net_g = SynthesizerTrn256(1025,32,192,192,768,2,6,3,0.1,"1", [3,7,11],[[1,3,5], [1,3,5], [1,3,5]],[10,10,2],512,[16,16,4],0)#hifigan-ps-sr
#
# net_g = SynthesizerTrn(1025, 32, 192, 192, 768, 2, 6, 3, 0.1, "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [5,5], 512, [15,15], 0)#ms
# net_g = SynthesizerTrn(1025, 32, 192, 192, 768, 2, 6, 3, 0.1, "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10,10], 512, [16,16], 0)#idwt2

# weights=torch.load("infer/ft-mi_1k-noD.pt")
# weights=torch.load("infer/ft-mi-freeze-vocoder-flow-enc_q_1k.pt")
# weights=torch.load("infer/ft-mi-freeze-vocoder_true_1k.pt")
# weights=torch.load("infer/ft-mi-sim1k.pt")
weights = torch.load("infer/ft-mi-no_opt-no_dropout.pt")
logger.debug(net_g.load_state_dict(weights, strict=True))

net_g.eval().to(device)
net_g.half()


def get_f0(x, p_len, f0_up_key=0):
    time_step = 160 / 16000 * 1000
    f0_min = 50
    f0_max = 1100
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)

    f0 = (
        parselmouth.Sound(x, 16000)
        .to_pitch_ac(
            time_step=time_step / 1000,
            voicing_threshold=0.6,
            pitch_floor=f0_min,
            pitch_ceiling=f0_max,
        )
        .selected_array["frequency"]
    )

    pad_size = (p_len - len(f0) + 1) // 2
    if pad_size > 0 or p_len - len(f0) - pad_size > 0:
        f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant")
    f0 *= pow(2, f0_up_key / 12)
    f0bak = f0.copy()

    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
        f0_mel_max - f0_mel_min
    ) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    # f0_mel[f0_mel > 188] = 188
    f0_coarse = np.rint(f0_mel).astype(np.int32)
    return f0_coarse, f0bak


import faiss

index = faiss.read_index("infer/added_IVF512_Flat_mi_baseline_src_feat.index")
big_npy = np.load("infer/big_src_feature_mi.npy")
ta0 = ta1 = ta2 = 0
for idx, name in enumerate(
    [
        "冬之花clip1.wav",
    ]
):  ##
    wav_path = "todo-songs/%s" % name  #
    f0_up_key = -2  #
    audio, sampling_rate = sf.read(wav_path)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio.transpose(1, 0))
    if sampling_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)

    feats = torch.from_numpy(audio).float()
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    feats = feats.view(1, -1)
    padding_mask = torch.BoolTensor(feats.shape).fill_(False)
    inputs = {
        "source": feats.half().to(device),
        "padding_mask": padding_mask.to(device),
        "output_layer": 9,  # layer 9
    }
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = ttime()
    with torch.no_grad():
        logits = model.extract_features(**inputs)
        feats = model.final_proj(logits[0])

    ####索引优化
    npy = feats[0].cpu().numpy().astype("float32")
    D, I = index.search(npy, 1)
    feats = (
        torch.from_numpy(big_npy[I.squeeze()].astype("float16")).unsqueeze(0).to(device)
    )

    feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = ttime()
    # p_len = min(feats.shape[1],10000,pitch.shape[0])#太大了爆显存
    p_len = min(feats.shape[1], 10000)  #
    pitch, pitchf = get_f0(audio, p_len, f0_up_key)
    p_len = min(feats.shape[1], 10000, pitch.shape[0])  # 太大了爆显存
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t2 = ttime()
    feats = feats[:, :p_len, :]
    pitch = pitch[:p_len]
    pitchf = pitchf[:p_len]
    p_len = torch.LongTensor([p_len]).to(device)
    pitch = torch.LongTensor(pitch).unsqueeze(0).to(device)
    sid = torch.LongTensor([0]).to(device)
    pitchf = torch.FloatTensor(pitchf).unsqueeze(0).to(device)
    with torch.no_grad():
        audio = (
            net_g.infer(feats, p_len, pitch, pitchf, sid)[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )  # nsf
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t3 = ttime()
    ta0 += t1 - t0
    ta1 += t2 - t1
    ta2 += t3 - t2
    # wavfile.write("ft-mi_1k-index256-noD-%s.wav"%name, 40000, audio)##
    # wavfile.write("ft-mi-freeze-vocoder-flow-enc_q_1k-%s.wav"%name, 40000, audio)##
    # wavfile.write("ft-mi-sim1k-%s.wav"%name, 40000, audio)##
    wavfile.write("ft-mi-no_opt-no_dropout-%s.wav" % name, 40000, audio)  ##


logger.debug("%.2fs %.2fs %.2fs", ta0, ta1, ta2)  #
