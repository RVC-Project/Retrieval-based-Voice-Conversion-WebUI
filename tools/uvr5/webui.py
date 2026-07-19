import logging
import os
import traceback

import ffmpeg
import torch

from configs.config import Config, IS_GPU
from tools.uvr5.bsroformer import Roformer_Loader
from tools.uvr5.mdxnet import MDXNetDereverb
from tools.uvr5.vr import AudioPre, AudioPreDeEcho
from i18n.i18n import I18nAuto


logger = logging.getLogger(__name__)
i18n = I18nAuto()
config = Config()
weight_uvr5_root = os.getenv("weight_uvr5_root", "assets/uvr5_weights")


def clean_path(path):
    path = path or ""
    if path.endswith(("\\", "/")):
        path = path[:-1]
    return path.replace("/", os.sep).replace("\\", os.sep).strip(" '\n\"\u202a")


def uvr(model_name, inp_root, save_root_vocal, paths, save_root_ins, agg, format0):
    infos = []
    try:
        inp_root = clean_path(inp_root)
        save_root_vocal = clean_path(save_root_vocal)
        save_root_ins = clean_path(save_root_ins)
        is_hp3 = "HP3" in model_name
        if model_name == "onnx_dereverb_By_FoxJoy":
            if config.dml:
                providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
            elif IS_GPU:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
            pre_fun = MDXNetDereverb(15, providers)
        elif "roformer" in model_name.lower():
            pre_fun = Roformer_Loader(
                model_path=os.path.join(weight_uvr5_root, model_name + ".ckpt"),
                config_path=os.path.join(weight_uvr5_root, model_name + ".yaml"),
                device=config.device,
                is_half=config.is_half,
            )
            if not os.path.exists(
                os.path.join(weight_uvr5_root, model_name + ".yaml")
            ):
                infos.append(i18n("未找到Roformer模型配置文件，正在使用内置默认配置"))
                yield "\n".join(infos)
        else:
            func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
            pre_fun = func(
                agg=int(agg),
                model_path=os.path.join(weight_uvr5_root, model_name + ".pth"),
                device=config.device,
                is_half=config.is_half,
            )
        if inp_root:
            paths = [os.path.join(inp_root, name) for name in os.listdir(inp_root)]
        else:
            paths = [path.name for path in (paths or [])]
        for path in paths:
            inp_path = os.path.join(inp_root, path)
            if not os.path.isfile(inp_path):
                continue
            need_reformat = True
            done = False
            try:
                info = ffmpeg.probe(inp_path, cmd="ffprobe")
                if (
                    info["streams"][0]["channels"] == 2
                    and info["streams"][0]["sample_rate"] == "44100"
                ):
                    need_reformat = False
                    pre_fun._path_audio_(
                        inp_path,
                        save_root_ins,
                        save_root_vocal,
                        format0,
                        is_hp3,
                    )
                    done = True
            except:
                traceback.print_exc()
            if need_reformat:
                tmp_path = "%s/%s.reformatted.wav" % (
                    os.environ["TEMP"],
                    os.path.basename(inp_path),
                )
                os.system(
                    'ffmpeg -i "%s" -vn -acodec pcm_s16le -ac 2 -ar 44100 "%s" -y'
                    % (inp_path, tmp_path)
                )
                inp_path = tmp_path
            try:
                if not done:
                    pre_fun._path_audio_(
                        inp_path,
                        save_root_ins,
                        save_root_vocal,
                        format0,
                        is_hp3,
                    )
                infos.append(i18n("%s → 成功") % os.path.basename(inp_path))
                yield "\n".join(infos)
            except Exception:
                infos.append(
                    "%s → %s\n%s"
                    % (os.path.basename(inp_path), i18n("失败"), traceback.format_exc())
                )
                yield "\n".join(infos)
    except Exception:
        infos.append("%s\n%s" % (i18n("失败"), traceback.format_exc()))
        yield "\n".join(infos)
    finally:
        try:
            if model_name == "onnx_dereverb_By_FoxJoy":
                del pre_fun.pred.model
                del pre_fun.pred.model_
            else:
                del pre_fun.model
                del pre_fun
        except:
            traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Executed torch.cuda.empty_cache()")
    yield "\n".join(infos)
