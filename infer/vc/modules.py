import traceback
import logging

logger = logging.getLogger(__name__)

import numpy as np
import soundfile as sf
import torch
from io import BytesIO

from infer.audio import load_audio, wav2
from infer.module.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from infer.vc.pipeline import Pipeline
from infer.vc.utils import *
from i18n.i18n import I18nAuto
from tools.progress import batch_status, should_report


i18n = I18nAuto()


def inference_status(title, state, detail=""):
    lines = ["【%s】" % i18n(title), "%s：%s" % (i18n("状态"), i18n(state))]
    if detail:
        lines.extend(["", str(detail).strip()])
    return "\n".join(lines)


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
        logger.info("%s: %s", i18n("选择模型"), sid)

        to_return_protect0 = {
            "visible": self.if_f0 != 0,
            "value": (
                to_return_protect[0] if self.if_f0 != 0 and to_return_protect else 0.5
            ),
            "__type__": "update",
        }
        to_return_protect1 = {
            "visible": self.if_f0 != 0,
            "value": (
                to_return_protect[1] if self.if_f0 != 0 and to_return_protect else 0.33
            ),
            "__type__": "update",
        }

        if sid == "" or sid == []:
            if (
                self.hubert_model is not None
            ):  # 考虑到轮询, 需要加个判断看是否 sid 是由有模型切换到无模型的
                logger.info(i18n("清理模型缓存"))
                del (self.net_g, self.n_spk, self.hubert_model, self.tgt_sr)  # ,cpt
                self.hubert_model = self.net_g = self.n_spk = self.hubert_model = (
                    self.tgt_sr
                ) = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                ###楼下不这么折腾清理不干净
                self.if_f0 = self.cpt.get("f0", 1)
                self.version = self.cpt.get("version", "v1")
                if self.version == "v1":
                    if self.if_f0 == 1:
                        self.net_g = SynthesizerTrnMs256NSFsid(
                            *self.cpt["config"], is_half=self.config.is_half
                        )
                    else:
                        self.net_g = SynthesizerTrnMs256NSFsid_nono(*self.cpt["config"])
                elif self.version == "v2":
                    if self.if_f0 == 1:
                        self.net_g = SynthesizerTrnMs768NSFsid(
                            *self.cpt["config"], is_half=self.config.is_half
                        )
                    else:
                        self.net_g = SynthesizerTrnMs768NSFsid_nono(*self.cpt["config"])
                del self.net_g, self.cpt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return (
                {"visible": False, "__type__": "update"},
                {
                    "visible": True,
                    "value": to_return_protect0,
                    "__type__": "update",
                },
                {
                    "visible": True,
                    "value": to_return_protect1,
                    "__type__": "update",
                },
                "",
                "",
            )
        person = f'{os.getenv("weight_root")}/{sid}'
        logger.info("%s: %s", i18n("正在加载模型"), person)

        self.cpt = torch.load(person, map_location="cpu")
        self.tgt_sr = self.cpt["config"][-1]
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        self.if_f0 = self.cpt.get("f0", 1)
        self.version = self.cpt.get("version", "v1")

        synthesizer_class = {
            ("v1", 1): SynthesizerTrnMs256NSFsid,
            ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
            ("v2", 1): SynthesizerTrnMs768NSFsid,
            ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
        }

        self.net_g = synthesizer_class.get(
            (self.version, self.if_f0), SynthesizerTrnMs256NSFsid
        )(*self.cpt["config"], is_half=self.config.is_half)

        del self.net_g.enc_q

        self.net_g.load_state_dict(self.cpt["weight"], strict=False)
        self.net_g.eval().to(self.config.device)
        if self.config.is_half:
            self.net_g = self.net_g.half()
        else:
            self.net_g = self.net_g.float()

        self.pipeline = Pipeline(self.tgt_sr, self.config)
        n_spk = self.cpt["config"][-3]
        index = {"value": get_index_path_from_model(sid), "__type__": "update"}
        logger.info("%s: %s", i18n("选择索引"), index["value"])

        return (
            (
                {"visible": True, "maximum": n_spk, "__type__": "update"},
                to_return_protect0,
                to_return_protect1,
                index,
                index,
            )
            if to_return_protect
            else {"visible": True, "maximum": n_spk, "__type__": "update"}
        )

    def vc_single(
        self,
        sid,
        input_audio_path,
        f0_up_key,
        f0_method,
        file_index,
        index_rate,
        resample_sr,
        rms_mix_rate,
        protect,
    ):
        if input_audio_path is None:
            return inference_status("单次推理", "等待输入", i18n("请上传音频文件")), None
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
                    file_index.strip(" ")
                    .strip('"')
                    .strip("\n")
                    .strip('"')
                    .strip(" ")
                    .replace("trained", "added")
                )
            else:
                file_index = ""  # 防止小白写错，自动帮他替换掉

            audio_opt = self.pipeline.pipeline(
                self.hubert_model,
                self.net_g,
                sid,
                audio,
                times,
                f0_up_key,
                f0_method,
                file_index,
                index_rate,
                self.if_f0,
                self.tgt_sr,
                resample_sr,
                rms_mix_rate,
                self.version,
                protect,
            )
            if self.tgt_sr != resample_sr >= 16000:
                tgt_sr = resample_sr
            else:
                tgt_sr = self.tgt_sr
            index_info = (
                "%s：%s" % (i18n("索引"), file_index)
                if os.path.exists(file_index)
                else "%s：%s" % (i18n("索引"), i18n("未使用"))
            )
            return (
                inference_status(
                    "单次推理",
                    "成功",
                    "%s\n%s：%s %.2fs | F0 %.2fs | %s %.2fs"
                    % (
                        index_info,
                        i18n("耗时"),
                        i18n("特征"),
                        times[0],
                        times[1],
                        i18n("合成"),
                        times[2],
                    ),
                ),
                (tgt_sr, audio_opt),
            )
        except Exception:
            info = traceback.format_exc()
            logger.warning(info)
            return inference_status("单次推理", "失败", info), (None, None)

    def vc_multi(
        self,
        sid,
        dir_path,
        opt_root,
        paths,
        f0_up_key,
        f0_method,
        file_index,
        index_rate,
        resample_sr,
        rms_mix_rate,
        protect,
        format1,
    ):
        try:
            dir_path = (
                (dir_path or "")
                .strip(" ")
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip(" ")
            )  # 防止小白拷路径头尾带了空格和"和回车
            opt_root = (
                (opt_root or "")
                .strip(" ")
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip(" ")
            )
            if not opt_root:
                yield inference_status(
                    "批量推理", "等待输入", i18n("请填写输出文件夹路径")
                )
                return
            os.makedirs(opt_root, exist_ok=True)
            try:
                if dir_path != "":
                    paths = [
                        os.path.join(dir_path, name) for name in os.listdir(dir_path)
                    ]
                else:
                    paths = [path if isinstance(path, str) else path.name for path in (paths or [])]
            except Exception:
                traceback.print_exc()
                paths = [
                    path if isinstance(path, str) else path.name for path in (paths or [])
                ]
            total = len(paths)
            if total == 0:
                yield batch_status(i18n("批量推理"), 0, 0, 0, 0)
                return
            success = 0
            failed = 0
            failures = []
            for idx, path in enumerate(paths):
                item_failed = False
                info, opt = self.vc_single(
                    sid,
                    path,
                    f0_up_key,
                    f0_method,
                    file_index,
                    index_rate,
                    resample_sr,
                    rms_mix_rate,
                    protect,
                )
                if opt and opt[0] is not None and opt[1] is not None:
                    try:
                        tgt_sr, audio_opt = opt
                        if format1 in ["wav", "flac"]:
                            sf.write(
                                "%s/%s.%s"
                                % (
                                    opt_root,
                                    os.path.splitext(os.path.basename(path))[0],
                                    format1,
                                ),
                                audio_opt,
                                tgt_sr,
                            )
                        else:
                            path = "%s/%s.%s" % (
                                opt_root,
                                os.path.splitext(os.path.basename(path))[0],
                                format1,
                            )
                            with BytesIO() as wavf:
                                sf.write(wavf, audio_opt, tgt_sr, format="wav")
                                wavf.seek(0, 0)
                                with open(path, "wb") as outf:
                                    wav2(wavf, outf, format1)
                        success += 1
                    except Exception:
                        info = "%s\n%s" % (info, traceback.format_exc())
                        failed += 1
                        item_failed = True
                        failures.append("%s：%s" % (os.path.basename(path), info))
                else:
                    failed += 1
                    item_failed = True
                    failures.append("%s：%s" % (os.path.basename(path), info))
                if should_report(idx, total) or item_failed:
                    yield batch_status(
                        i18n("批量推理"),
                        idx + 1,
                        total,
                        success,
                        failed,
                        os.path.basename(path),
                        failures,
                    )
        except Exception:
            yield inference_status("批量推理", "失败", traceback.format_exc())
