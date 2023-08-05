import faiss, torch, traceback, parselmouth, numpy as np, torchcrepe, torch.nn as nn, pyworld
from fairseq import checkpoint_utils
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
import os, sys
from time import time as ttime
import torch.nn.functional as F
import scipy.signal as signal

now_dir = os.getcwd()
sys.path.append(now_dir)
from config import Config
from multiprocessing import Manager as M

mm = M()
config = Config()


class RVC:
    def __init__(
        self, key, pth_path, index_path, index_rate, n_cpu, inp_q, opt_q, device
    ) -> None:
        """
        初始化
        """
        try:
            global config
            self.inp_q = inp_q
            self.opt_q = opt_q
            self.device = device
            self.f0_up_key = key
            self.time_step = 160 / 16000 * 1000
            self.f0_min = 50
            self.f0_max = 1100
            self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
            self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
            self.sr = 16000
            self.window = 160
            self.n_cpu = n_cpu
            if index_rate != 0:
                self.index = faiss.read_index(index_path)
                self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
                print("index search enabled")
            self.index_rate = index_rate
            models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
                ["hubert_base.pt"],
                suffix="",
            )
            hubert_model = models[0]
            hubert_model = hubert_model.to(config.device)
            if config.is_half:
                hubert_model = hubert_model.half()
            else:
                hubert_model = hubert_model.float()
            hubert_model.eval()
            self.model = hubert_model
            cpt = torch.load(pth_path, map_location="cpu")
            self.tgt_sr = cpt["config"][-1]
            cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
            self.if_f0 = cpt.get("f0", 1)
            self.version = cpt.get("version", "v1")
            if self.version == "v1":
                if self.if_f0 == 1:
                    self.net_g = SynthesizerTrnMs256NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    self.net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
            elif self.version == "v2":
                if self.if_f0 == 1:
                    self.net_g = SynthesizerTrnMs768NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    self.net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
            del self.net_g.enc_q
            print(self.net_g.load_state_dict(cpt["weight"], strict=False))
            self.net_g.eval().to(device)
            if config.is_half:
                self.net_g = self.net_g.half()
            else:
                self.net_g = self.net_g.float()
            self.is_half = config.is_half
        except:
            print(traceback.format_exc())

    def get_f0_post(self, f0):
        f0_min = self.f0_min
        f0_max = self.f0_max
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int32)
        return f0_coarse, f0bak

    def get_f0(self, x, f0_up_key, n_cpu, method="harvest"):
        n_cpu = int(n_cpu)
        if method == "crepe":
            return self.get_f0_crepe(x, f0_up_key)
        if method == "rmvpe":
            return self.get_f0_rmvpe(x, f0_up_key)
        if method == "pm":
            p_len = x.shape[0] // 160
            f0 = (
                parselmouth.Sound(x, 16000)
                .to_pitch_ac(
                    time_step=0.01,
                    voicing_threshold=0.6,
                    pitch_floor=50,
                    pitch_ceiling=1100,
                )
                .selected_array["frequency"]
            )

            pad_size = (p_len - len(f0) + 1) // 2
            if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                print(pad_size, p_len - len(f0) - pad_size)
                f0 = np.pad(
                    f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
                )

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
        f0bak = np.zeros(x.shape[0] // 160, dtype=np.float64)
        length = len(x)
        part_length = int(length / n_cpu / 160) * 160
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
                f0 = f0[2:-1]
            f0bak[
                part_length * idx // 160 : part_length * idx // 160 + f0.shape[0]
            ] = f0
        f0bak = signal.medfilt(f0bak, 3)
        f0bak *= pow(2, f0_up_key / 12)
        return self.get_f0_post(f0bak)

    def get_f0_crepe(self, x, f0_up_key):
        audio = torch.tensor(np.copy(x))[None].float()
        f0, pd = torchcrepe.predict(
            audio,
            self.sr,
            160,
            self.f0_min,
            self.f0_max,
            "full",
            batch_size=512,
            device=self.device,
            return_periodicity=True,
        )
        pd = torchcrepe.filter.median(pd, 3)
        f0 = torchcrepe.filter.mean(f0, 3)
        f0[pd < 0.1] = 0
        f0 = f0[0].cpu().numpy()
        f0 *= pow(2, f0_up_key / 12)
        return self.get_f0_post(f0)

    def get_f0_rmvpe(self, x, f0_up_key):
        if hasattr(self, "model_rmvpe") == False:
            from lib.rmvpe import RMVPE

            print("loading rmvpe model")
            self.model_rmvpe = RMVPE(
                "rmvpe.pt", is_half=self.is_half, device=self.device
            )
            # self.model_rmvpe = RMVPE("aug2_58000_half.pt", is_half=self.is_half, device=self.device)
        f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        f0 *= pow(2, f0_up_key / 12)
        return self.get_f0_post(f0)

    def infer(
        self,
        feats: torch.Tensor,
        indata: np.ndarray,
        rate1,
        rate2,
        cache_pitch,
        cache_pitchf,
        f0method,
    ) -> np.ndarray:
        feats = feats.view(1, -1)
        if config.is_half:
            feats = feats.half()
        else:
            feats = feats.float()
        feats = feats.to(self.device)
        t1 = ttime()
        with torch.no_grad():
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
        t2 = ttime()
        try:
            if hasattr(self, "index") and self.index_rate != 0:
                leng_replace_head = int(rate1 * feats[0].shape[0])
                npy = feats[0][-leng_replace_head:].cpu().numpy().astype("float32")
                score, ix = self.index.search(npy, k=8)
                weight = np.square(1 / score)
                weight /= weight.sum(axis=1, keepdims=True)
                npy = np.sum(self.big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
                if config.is_half:
                    npy = npy.astype("float16")
                feats[0][-leng_replace_head:] = (
                    torch.from_numpy(npy).unsqueeze(0).to(self.device) * self.index_rate
                    + (1 - self.index_rate) * feats[0][-leng_replace_head:]
                )
            else:
                print("index search FAIL or disabled")
        except:
            traceback.print_exc()
            print("index search FAIL")
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        t3 = ttime()
        if self.if_f0 == 1:
            pitch, pitchf = self.get_f0(indata, self.f0_up_key, self.n_cpu, f0method)
            cache_pitch[:] = np.append(cache_pitch[pitch[:-1].shape[0] :], pitch[:-1])
            cache_pitchf[:] = np.append(
                cache_pitchf[pitchf[:-1].shape[0] :], pitchf[:-1]
            )
            p_len = min(feats.shape[1], 13000, cache_pitch.shape[0])
        else:
            cache_pitch, cache_pitchf = None, None
            p_len = min(feats.shape[1], 13000)
        t4 = ttime()
        feats = feats[:, :p_len, :]
        if self.if_f0 == 1:
            cache_pitch = cache_pitch[:p_len]
            cache_pitchf = cache_pitchf[:p_len]
            cache_pitch = torch.LongTensor(cache_pitch).unsqueeze(0).to(self.device)
            cache_pitchf = torch.FloatTensor(cache_pitchf).unsqueeze(0).to(self.device)
        p_len = torch.LongTensor([p_len]).to(self.device)
        ii = 0  # sid
        sid = torch.LongTensor([ii]).to(self.device)
        with torch.no_grad():
            if self.if_f0 == 1:
                infered_audio = (
                    self.net_g.infer(
                        feats, p_len, cache_pitch, cache_pitchf, sid, rate2
                    )[0][0, 0]
                    .data.cpu()
                    .float()
                )
            else:
                infered_audio = (
                    self.net_g.infer(feats, p_len, sid, rate2)[0][0, 0]
                    .data.cpu()
                    .float()
                )
        t5 = ttime()
        print("time->fea-index-f0-model:", t2 - t1, t3 - t2, t4 - t3, t5 - t4)
        return infered_audio
