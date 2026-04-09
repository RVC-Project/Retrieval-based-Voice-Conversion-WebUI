import traceback
import logging

logger = logging.getLogger(__name__)

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from io import BytesIO

from infer.lib.audio import load_audio, wav2
from infer.lib.infer_pack.models import (
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from infer.modules.vc.pipeline import Pipeline
from infer.modules.vc.utils import *


class VC:
    def __init__(self, config):
        self.n_spk = None
        self.tgt_sr = None
        self.net_g = None
        self.pipeline = None
        self.cpt = None
        self.version = None
        self.if_f0 = None
        self.version = None
        self.hubert_model = None

        self.config = config

    def get_vc(self, sid, *to_return_protect):
        logger.info("Get sid: " + sid)

        to_return_protect0 = gr.update(
            visible=self.if_f0 != 0,
            value=(to_return_protect[0] if self.if_f0 != 0 and to_return_protect else 0.5),
        )
        to_return_protect1 = gr.update(
            visible=self.if_f0 != 0,
            value=(to_return_protect[1] if self.if_f0 != 0 and to_return_protect else 0.33),
        )

        if sid == "" or sid == []:
            if self.hubert_model is not None:  # 考虑到轮询, 需要加个判断看是否 sid 是由有模型切换到无模型的
                logger.info("Clean model cache")
                del (self.net_g, self.n_spk, self.hubert_model, self.tgt_sr)  # ,cpt
                self.hubert_model = self.net_g = self.n_spk = self.hubert_model = self.tgt_sr = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                ###楼下不这么折腾清理不干净
                self.if_f0 = self.cpt.get("f0", 1)
                self.version = self.cpt.get("version", "v2")
                if self.if_f0 == 1:
                    self.net_g = SynthesizerTrnMs768NSFsid(*self.cpt["config"], is_half=self.config.is_half)
                else:
                    self.net_g = SynthesizerTrnMs768NSFsid_nono(*self.cpt["config"])
                del self.net_g, self.cpt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return (
                gr.update(visible=False),
                gr.update(visible=True, value=to_return_protect0),
                gr.update(visible=True, value=to_return_protect1),
                "",
                "",
            )
        person = f"{os.getenv('weight_root')}/{sid}"
        logger.info(f"Loading: {person}")

        self.cpt = torch.load(person, map_location="cpu", weights_only=False)
        self.tgt_sr = self.cpt["config"][-1]
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        self.if_f0 = self.cpt.get("f0", 1)
        self.version = self.cpt.get("version", "v2")

        if self.if_f0:
            self.net_g = SynthesizerTrnMs768NSFsid(*self.cpt["config"], is_half=self.config.is_half)
        else:
            self.net_g = SynthesizerTrnMs768NSFsid_nono(*self.cpt["config"])

        del self.net_g.enc_q

        self.net_g.load_state_dict(self.cpt["weight"], strict=False)
        self.net_g.eval().to(self.config.device)
        if self.config.is_half:
            self.net_g = self.net_g.half()
        else:
            self.net_g = self.net_g.float()

        self.pipeline = Pipeline(self.tgt_sr, self.config)
        n_spk = self.cpt["config"][-3]
        index_value = get_index_path_from_model(sid)
        index = gr.update(value=index_value)
        logger.info("Select index: " + index_value)

        return (
            (
                gr.update(visible=True, maximum=n_spk),
                to_return_protect0,
                to_return_protect1,
                index,
                index,
            )
            if to_return_protect
            else gr.update(visible=True, maximum=n_spk)
        )

    def vc_single(
        self,
        sid,
        input_audio_path,
        f0_up_key,
        f0_file,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
    ):
        # デバッグログ: 入力値の型と内容を確認
        logger.info(f"vc_single called with input_audio_path type: {type(input_audio_path)}, value: {input_audio_path}")

        if input_audio_path is None:
            return "You need to upload an audio", None

        # gr.Audio が辞書やタプルを返す場合の対応
        if isinstance(input_audio_path, dict):
            logger.info(f"input_audio_path is dict with keys: {input_audio_path.keys()}")
            input_audio_path = input_audio_path.get("name") or input_audio_path.get("path")
        elif isinstance(input_audio_path, (list, tuple)):
            logger.info(f"input_audio_path is list/tuple with length: {len(input_audio_path)}")
            input_audio_path = input_audio_path[0] if input_audio_path else None

        logger.info(f"After extraction, input_audio_path: {input_audio_path}")

        if input_audio_path is None:
            return "You need to upload an audio", None

        f0_up_key = int(f0_up_key)
        try:
            audio = load_audio(input_audio_path, 16000)
            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max
            times = [0, 0, 0]

            if self.hubert_model is None:
                self.hubert_model = load_hubert(self.config)

            if file_index:
                file_index = (
                    file_index.strip(" ").strip('"').strip("\n").strip('"').strip(" ").replace("trained", "added")
                )
            elif file_index2:
                file_index = file_index2
            else:
                file_index = ""  # 防止小白写错，自动帮他替换掉

            audio_opt = self.pipeline.pipeline(
                self.hubert_model,
                self.net_g,
                sid,
                audio,
                input_audio_path,
                times,
                f0_up_key,
                f0_method,
                file_index,
                index_rate,
                self.if_f0,
                filter_radius,
                self.tgt_sr,
                resample_sr,
                rms_mix_rate,
                self.version,
                protect,
                f0_file,
            )
            if self.tgt_sr != resample_sr >= 16000:
                tgt_sr = resample_sr
            else:
                tgt_sr = self.tgt_sr
            index_info = "Index:\n%s." % file_index if os.path.exists(file_index) else "Index not used."
            return (
                "Success.\n%s\nTime:\nnpy: %.2fs, f0: %.2fs, infer: %.2fs." % (index_info, *times),
                (tgt_sr, audio_opt),
            )
        except Exception:
            info = traceback.format_exc()
            logger.warning(info)
            return info, (None, None)

    def vc_multi(
        self,
        sid,
        dir_path,
        opt_root,
        paths,
        f0_up_key,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
        format1,
    ):
        try:
            dir_path = (
                dir_path.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            )  # 防止小白拷路径头尾带了空格和"和回车
            opt_root = opt_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            os.makedirs(opt_root, exist_ok=True)
            try:
                if dir_path != "":
                    paths = [os.path.join(dir_path, name) for name in os.listdir(dir_path)]
                else:
                    paths = [p if isinstance(p, str) else p.name for p in paths]
            except Exception:
                traceback.print_exc()
                paths = [p if isinstance(p, str) else p.name for p in paths]
            infos = []
            for path in paths:
                info, opt = self.vc_single(
                    sid,
                    path,
                    f0_up_key,
                    None,
                    f0_method,
                    file_index,
                    file_index2,
                    # file_big_npy,
                    index_rate,
                    filter_radius,
                    resample_sr,
                    rms_mix_rate,
                    protect,
                )
                if "Success" in info:
                    try:
                        tgt_sr, audio_opt = opt
                        if format1 in ["wav", "flac"]:
                            sf.write(
                                "%s/%s.%s" % (opt_root, os.path.basename(path), format1),
                                audio_opt,
                                tgt_sr,
                            )
                        else:
                            path = "%s/%s.%s" % (
                                opt_root,
                                os.path.basename(path),
                                format1,
                            )
                            with BytesIO() as wavf:
                                sf.write(wavf, audio_opt, tgt_sr, format="wav")
                                wavf.seek(0, 0)
                                with open(path, "wb") as outf:
                                    wav2(wavf, outf, format1)
                    except Exception:
                        info += traceback.format_exc()
                infos.append("%s->%s" % (os.path.basename(path), info))
                yield "\n".join(infos)
            yield "\n".join(infos)
        except Exception:
            yield traceback.format_exc()
