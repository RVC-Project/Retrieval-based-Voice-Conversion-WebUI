from io import BytesIO
import os
import pickle
import sys
import traceback
from infer.lib import jit
from infer.lib.jit.get_synthesizer import get_synthesizer
from time import time as ttime
import fairseq
import faiss
import numpy as np
import parselmouth
import pyworld
import scipy.signal as signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchcrepe

from infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)

now_dir = os.getcwd()
sys.path.append(now_dir)
from multiprocessing import Manager as M

from configs.config import Config

# config = Config()

mm = M()


def printt(strr, *args):
    if len(args) == 0:
        print(strr)
    else:
        print(strr % args)


# config.device=torch.device("cpu")########强制cpu测试
# config.is_half=False########强制cpu测试
class RVC:
    def __init__(
        self,
        key,
        pth_path,
        index_path,
        index_rate,
        n_cpu,
        inp_q,
        opt_q,
        config: Config,
        last_rvc=None,
    ) -> None:
        """
        初始化
        """
        try:
            if config.dml == True:

                def forward_dml(ctx, x, scale):
                    ctx.scale = scale
                    res = x.clone().detach()
                    return res

                fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml
            # global config
            self.config = config
            self.inp_q = inp_q
            self.opt_q = opt_q
            # device="cpu"########强制cpu测试
            self.device = config.device
            self.f0_up_key = key
            self.f0_min = 50
            self.f0_max = 1100
            self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
            self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
            self.n_cpu = n_cpu
            self.use_jit = self.config.use_jit
            self.is_half = config.is_half

            if index_rate != 0:
                self.index = faiss.read_index(index_path)
                self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
                printt("Index search enabled")
            self.pth_path: str = pth_path
            self.index_path = index_path
            self.index_rate = index_rate
            self.cache_pitch: torch.Tensor = torch.zeros(
                1024, device=self.device, dtype=torch.long
            )
            self.cache_pitchf = torch.zeros(
                1024, device=self.device, dtype=torch.float32
            )

            if last_rvc is None:
                models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                    ["assets/hubert/hubert_base.pt"],
                    suffix="",
                )
                hubert_model = models[0]
                hubert_model = hubert_model.to(self.device)
                if self.is_half:
                    hubert_model = hubert_model.half()
                else:
                    hubert_model = hubert_model.float()
                hubert_model.eval()
                self.model = hubert_model
            else:
                self.model = last_rvc.model

            self.net_g: nn.Module = None

            def set_default_model():
                self.net_g, cpt = get_synthesizer(self.pth_path, self.device)
                self.tgt_sr = cpt["config"][-1]
                cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
                self.if_f0 = cpt.get("f0", 1)
                self.version = cpt.get("version", "v1")
                if self.is_half:
                    self.net_g = self.net_g.half()
                else:
                    self.net_g = self.net_g.float()

            def set_jit_model():
                jit_pth_path = self.pth_path.rstrip(".pth")
                jit_pth_path += ".half.jit" if self.is_half else ".jit"
                reload = False
                if str(self.device) == "cuda":
                    self.device = torch.device("cuda:0")
                if os.path.exists(jit_pth_path):
                    cpt = jit.load(jit_pth_path)
                    model_device = cpt["device"]
                    if model_device != str(self.device):
                        reload = True
                else:
                    reload = True

                if reload:
                    cpt = jit.synthesizer_jit_export(
                        self.pth_path,
                        "script",
                        None,
                        device=self.device,
                        is_half=self.is_half,
                    )

                self.tgt_sr = cpt["config"][-1]
                self.if_f0 = cpt.get("f0", 1)
                self.version = cpt.get("version", "v1")
                self.net_g = torch.jit.load(
                    BytesIO(cpt["model"]), map_location=self.device
                )
                self.net_g.infer = self.net_g.forward
                self.net_g.eval().to(self.device)

            def set_synthesizer():
                if self.use_jit and not config.dml:
                    if self.is_half and "cpu" in str(self.device):
                        printt(
                            "Use default Synthesizer model. \
                                    Jit is not supported on the CPU for half floating point"
                        )
                        set_default_model()
                    else:
                        set_jit_model()
                else:
                    set_default_model()

            if last_rvc is None or last_rvc.pth_path != self.pth_path:
                set_synthesizer()
            else:
                self.tgt_sr = last_rvc.tgt_sr
                self.if_f0 = last_rvc.if_f0
                self.version = last_rvc.version
                self.is_half = last_rvc.is_half
                if last_rvc.use_jit != self.use_jit:
                    set_synthesizer()
                else:
                    self.net_g = last_rvc.net_g

            if last_rvc is not None and hasattr(last_rvc, "model_rmvpe"):
                self.model_rmvpe = last_rvc.model_rmvpe
            if last_rvc is not None and hasattr(last_rvc, "model_fcpe"):
                self.device_fcpe = last_rvc.device_fcpe
                self.model_fcpe = last_rvc.model_fcpe
        except:
            printt(traceback.format_exc())

    def change_key(self, new_key):
        self.f0_up_key = new_key

    def change_index_rate(self, new_index_rate):
        if new_index_rate != 0 and self.index_rate == 0:
            self.index = faiss.read_index(self.index_path)
            self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
            printt("Index search enabled")
        self.index_rate = new_index_rate

    def get_f0_post(self, f0):
        if not torch.is_tensor(f0):
            f0 = torch.from_numpy(f0)
        f0 = f0.float().to(self.device).squeeze()
        f0_mel = 1127 * torch.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (
            self.f0_mel_max - self.f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = torch.round(f0_mel).long()
        return f0_coarse, f0

    def get_f0(self, x, f0_up_key, n_cpu, method="harvest"):
        n_cpu = int(n_cpu)
        if method == "crepe":
            return self.get_f0_crepe(x, f0_up_key)
        if method == "rmvpe":
            return self.get_f0_rmvpe(x, f0_up_key)
        if method == "fcpe":
            return self.get_f0_fcpe(x, f0_up_key)
        x = x.cpu().numpy()
        if method == "pm":
            p_len = x.shape[0] // 160 + 1
            f0_min = 65
            l_pad = int(np.ceil(1.5 / f0_min * 16000))
            r_pad = l_pad + 1
            s = parselmouth.Sound(np.pad(x, (l_pad, r_pad)), 16000).to_pitch_ac(
                time_step=0.01,
                voicing_threshold=0.6,
                pitch_floor=f0_min,
                pitch_ceiling=1100,
            )
            assert np.abs(s.t1 - 1.5 / f0_min) < 0.001
            f0 = s.selected_array["frequency"]
            if len(f0) < p_len:
                f0 = np.pad(f0, (0, p_len - len(f0)))
            f0 = f0[:p_len]
            f0 *= pow(2, f0_up_key / 12)
            return self.get_f0_post(f0)
        if n_cpu == 1:
            f0, t = pyworld.harvest(
                x.astype(np.double),
                fs=16000,
                f0_ceil=1100,
                f0_floor=50,
                frame_period=10,
            )
            f0 = signal.medfilt(f0, 3)
            f0 *= pow(2, f0_up_key / 12)
            return self.get_f0_post(f0)
        f0bak = np.zeros(x.shape[0] // 160 + 1, dtype=np.float64)
        length = len(x)
        part_length = 160 * ((length // 160 - 1) // n_cpu + 1)
        n_cpu = (length // 160 - 1) // (part_length // 160) + 1
        ts = ttime()
        res_f0 = mm.dict()
        for idx in range(n_cpu):
            tail = part_length * (idx + 1) + 320
            if idx == 0:
                self.inp_q.put((idx, x[:tail], res_f0, n_cpu, ts))
            else:
                self.inp_q.put(
                    (idx, x[part_length * idx - 320 : tail], res_f0, n_cpu, ts)
                )
        while 1:
            res_ts = self.opt_q.get()
            if res_ts == ts:
                break
        f0s = [i[1] for i in sorted(res_f0.items(), key=lambda x: x[0])]
        for idx, f0 in enumerate(f0s):
            if idx == 0:
                f0 = f0[:-3]
            elif idx != n_cpu - 1:
                f0 = f0[2:-3]
            else:
                f0 = f0[2:]
            f0bak[part_length * idx // 160 : part_length * idx // 160 + f0.shape[0]] = (
                f0
            )
        f0bak = signal.medfilt(f0bak, 3)
        f0bak *= pow(2, f0_up_key / 12)
        return self.get_f0_post(f0bak)

    def get_f0_crepe(self, x, f0_up_key):
        if "privateuseone" in str(
            self.device
        ):  ###不支持dml，cpu又太慢用不成，拿fcpe顶替
            return self.get_f0(x, f0_up_key, 1, "fcpe")
        # printt("using crepe,device:%s"%self.device)
        f0, pd = torchcrepe.predict(
            x.unsqueeze(0).float(),
            16000,
            160,
            self.f0_min,
            self.f0_max,
            "full",
            batch_size=512,
            # device=self.device if self.device.type!="privateuseone" else "cpu",###crepe不用半精度全部是全精度所以不愁###cpu延迟高到没法用
            device=self.device,
            return_periodicity=True,
        )
        pd = torchcrepe.filter.median(pd, 3)
        f0 = torchcrepe.filter.mean(f0, 3)
        f0[pd < 0.1] = 0
        f0 *= pow(2, f0_up_key / 12)
        return self.get_f0_post(f0)

    def get_f0_rmvpe(self, x, f0_up_key):
        if hasattr(self, "model_rmvpe") == False:
            from infer.lib.rmvpe import RMVPE

            printt("Loading rmvpe model")
            self.model_rmvpe = RMVPE(
                "assets/rmvpe/rmvpe.pt",
                is_half=self.is_half,
                device=self.device,
                use_jit=self.config.use_jit,
            )
        f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        f0 *= pow(2, f0_up_key / 12)
        return self.get_f0_post(f0)

    def get_f0_fcpe(self, x, f0_up_key):
        if hasattr(self, "model_fcpe") == False:
            from torchfcpe import spawn_bundled_infer_model

            printt("Loading fcpe model")
            if "privateuseone" in str(self.device):
                self.device_fcpe = "cpu"
            else:
                self.device_fcpe = self.device
            self.model_fcpe = spawn_bundled_infer_model(self.device_fcpe)
        f0 = self.model_fcpe.infer(
            x.to(self.device_fcpe).unsqueeze(0).float(),
            sr=16000,
            decoder_mode="local_argmax",
            threshold=0.006,
        )
        f0 *= pow(2, f0_up_key / 12)
        return self.get_f0_post(f0)

    def infer(
        self,
        input_wav: torch.Tensor,
        block_frame_16k,
        skip_head,
        return_length,
        f0method,
    ) -> np.ndarray:
        t1 = ttime()
        with torch.no_grad():
            if self.config.is_half:
                feats = input_wav.half().view(1, -1)
            else:
                feats = input_wav.float().view(1, -1)
            padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)
            inputs = {
                "source": feats,
                "padding_mask": padding_mask,
                "output_layer": 9 if self.version == "v1" else 12,
            }
            logits = self.model.extract_features(**inputs)
            feats = (
                self.model.final_proj(logits[0]) if self.version == "v1" else logits[0]
            )
            feats = torch.cat((feats, feats[:, -1:, :]), 1)
        t2 = ttime()
        try:
            if hasattr(self, "index") and self.index_rate != 0:
                npy = feats[0][skip_head // 2 :].cpu().numpy().astype("float32")
                score, ix = self.index.search(npy, k=8)
                if (ix >= 0).all():
                    weight = np.square(1 / score)
                    weight /= weight.sum(axis=1, keepdims=True)
                    npy = np.sum(
                        self.big_npy[ix] * np.expand_dims(weight, axis=2), axis=1
                    )
                    if self.config.is_half:
                        npy = npy.astype("float16")
                    feats[0][skip_head // 2 :] = (
                        torch.from_numpy(npy).unsqueeze(0).to(self.device)
                        * self.index_rate
                        + (1 - self.index_rate) * feats[0][skip_head // 2 :]
                    )
                else:
                    printt(
                        "Invalid index. You MUST use added_xxxx.index but not trained_xxxx.index!"
                    )
            else:
                printt("Index search FAILED or disabled")
        except:
            traceback.print_exc()
            printt("Index search FAILED")
        t3 = ttime()
        p_len = input_wav.shape[0] // 160
        if self.if_f0 == 1:
            f0_extractor_frame = block_frame_16k + 800
            if f0method == "rmvpe":
                f0_extractor_frame = 5120 * ((f0_extractor_frame - 1) // 5120 + 1) - 160
            pitch, pitchf = self.get_f0(
                input_wav[-f0_extractor_frame:], self.f0_up_key, self.n_cpu, f0method
            )
            shift = block_frame_16k // 160
            self.cache_pitch[:-shift] = self.cache_pitch[shift:].clone()
            self.cache_pitchf[:-shift] = self.cache_pitchf[shift:].clone()
            self.cache_pitch[4 - pitch.shape[0] :] = pitch[3:-1]
            self.cache_pitchf[4 - pitch.shape[0] :] = pitchf[3:-1]
            cache_pitch = self.cache_pitch[None, -p_len:]
            cache_pitchf = self.cache_pitchf[None, -p_len:]
        t4 = ttime()
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        feats = feats[:, :p_len, :]
        p_len = torch.LongTensor([p_len]).to(self.device)
        sid = torch.LongTensor([0]).to(self.device)
        skip_head = torch.LongTensor([skip_head])
        return_length = torch.LongTensor([return_length])
        with torch.no_grad():
            if self.if_f0 == 1:
                infered_audio, _, _ = self.net_g.infer(
                    feats,
                    p_len,
                    cache_pitch,
                    cache_pitchf,
                    sid,
                    skip_head,
                    return_length,
                )
            else:
                infered_audio, _, _ = self.net_g.infer(
                    feats, p_len, sid, skip_head, return_length
                )
        t5 = ttime()
        printt(
            "Spent time: fea = %.3fs, index = %.3fs, f0 = %.3fs, model = %.3fs",
            t2 - t1,
            t3 - t2,
            t4 - t3,
            t5 - t4,
        )
        return infered_audio.squeeze().float()
