import os

from vc_infer_pipeline import VC
from lib.audio import load_audio
import soundfile as sf
import fairseq
import torch
import traceback
import numpy as np
from src.configuration import *
from scipy.io import wavfile
import json
import taglib
from datetime import datetime

from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)


hubert_model = None

def load_hubert():
    global hubert_model
    models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
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



def get_index_path_from_model(sid):
    sel_index_path = ""
    name = os.path.join("logs", sid.split(".")[0], "")
    # print(name)
    for f in index_paths:
        if name in f:
            # print("selected index path:", f)
            sel_index_path = f
            break
    return sel_index_path


# 一个选项卡全局只能有一个音色
def get_vc(sid, to_return_protect0, to_return_protect1):
    global n_spk, tgt_sr, net_g, vc, cpt, version
    if sid == "" or sid == []:
        global hubert_model
        if hubert_model is not None:  # 考虑到轮询, 需要加个判断看是否 sid 是由有模型切换到无模型的
            print("clean_empty_cache")
            del net_g, n_spk, vc, hubert_model, tgt_sr  # ,cpt
            hubert_model = net_g = n_spk = vc = hubert_model = tgt_sr = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            ###楼下不这么折腾清理不干净
            if_f0 = cpt.get("f0", 1)
            version = cpt.get("version", "v1")
            if version == "v1":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs256NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
            elif version == "v2":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs768NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
            del net_g, cpt
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
    person = "%s/%s" % (weight_root, sid)
    print("loading %s" % person)

    cpt = torch.load(person, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    if if_f0 == 0:
        to_return_protect0 = to_return_protect1 = {
            "visible": False,
            "value": 0.33,
            "__type__": "update",
        }
    else:
        to_return_protect0 = {
            "visible": True,
            "value": to_return_protect0,
            "__type__": "update",
        }
        to_return_protect1 = {
            "visible": True,
            "value": to_return_protect1,
            "__type__": "update",
        }
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]
    index = {"value": get_index_path_from_model(sid), "__type__": "update"}
    return (
        {"visible": True, "maximum": n_spk, "__type__": "update"},
        to_return_protect0,
        to_return_protect1,
        index,
        index,
    )


def vc_single(
    sid=0,
    input_audio_path=None,
    f0_up_key=0,
    f0_file=None,
    f0_method="pm",
    file_index="",  # .index file
    file_index2="",
    # file_big_npy,
    index_rate=1.0,
    filter_radius=3,
    resample_sr=0,
    rms_mix_rate=1.0,
    protect=0.33,
):  # spk_item, input_audio0, vc_transform0,f0_file,f0method0
    global tgt_sr, net_g, vc, hubert_model, version, cpt
    if input_audio_path is None:
        return "You need to upload an audio", None
    f0_up_key = int(f0_up_key)
    try:
        audio = load_audio(input_audio_path, 16000)
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max
        times = [0, 0, 0]
        if not hubert_model:
            load_hubert()
        if_f0 = cpt.get("f0", 1)
        file_index = (
            (
                file_index.strip(" ")
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip(" ")
                .replace("trained", "added")
            )
            if file_index != ""
            else file_index2
        )  # 防止小白写错，自动帮他替换掉
        # file_big_npy = (
        #     file_big_npy.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        # )
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            sid,
            audio,
            input_audio_path,
            times,
            f0_up_key,
            f0_method,
            file_index,
            # file_big_npy,
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
        index_info = (
            "Using index:%s." % file_index
            if os.path.exists(file_index)
            else "Index not used."
        )
        return "Success.\n %s\nTime:\n npy:%ss, f0:%ss, infer:%ss" % (
            index_info,
            times[0],
            times[1],
            times[2],
        ), (resample_sr if resample_sr >= 16000 and tgt_sr != resample_sr else tgt_sr, audio_opt)
    except:
        info = traceback.format_exc()
        print(info)
        return info, (None, None)


def vc_multi(
    sid,
    dir_path,
    opt_root,
    paths,
    f0_up_key,
    f0_method,
    file_index,
    file_index2,
    # file_big_npy,
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
                paths = [path.name for path in paths]
        except:
            traceback.print_exc()
            paths = [path.name for path in paths]
        infos = []
        for path in paths:
            info, opt = vc_single(
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
                        path = "%s/%s.wav" % (opt_root, os.path.basename(path))
                        sf.write(
                            path,
                            audio_opt,
                            tgt_sr,
                        )
                        if os.path.exists(path):
                            os.system(
                                "ffmpeg -i %s -vn %s -q:a 2 -y"
                                % (path, path[:-4] + ".%s" % format1)
                            )
                except:
                    info += traceback.format_exc()
            infos.append("%s->%s" % (os.path.basename(path), info))
            yield "\n".join(infos)
        yield "\n".join(infos)
    except:
        yield traceback.format_exc()


def get_vc_from_model_path(model_path):
    global n_spk, tgt_sr, net_g, vc, cpt, config, version
    print("loading pth %s" % model_path)
    cpt = torch.load(model_path, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]
    # return {"visible": True,"maximum": n_spk, "__type__": "update"}


def vc_single_json(data):
    if not data.get('input_audio_path'):
        return {
            'info': "no input file specified",
            'error': "no input file specified"
        }

    # load model if provided
    if data.get('model_path'):
        get_vc_from_model_path(data['model_path'])

    # do conversion
    result = vc_single(
        data.get('sid', 0),
        data['input_audio_path'],
        data.get('f0_up_key', 0),
        data.get('f0_file', None),
        data.get('f0_method', 'pm'),
        data.get('file_index', ""),
        data.get('file_index2', ""),
        data.get('index_rate', 1.0),
        data.get('filter_radius', 3),
        data.get('resample_sr', 0),
        data.get('rms_mix_rate', 1),
        data.get('protect', 0.33))

    resultPretty = { 'info': result[0], 'sample_rate': result[1][0], 'audio': result[1][1]}

    if not isinstance(resultPretty['audio'], np.ndarray):
        return {
            "info": resultPretty['info'],
            "error": "no audio generated"
        }

    # optional write to wave file
    if data.get('output_path'):

        def AddFileSuffix(filename, suffix):
            filename_base = os.path.splitext(os.path.basename(filename))[0]
            filename_extension = os.path.splitext(os.path.basename(filename))[-1][1:]

            return os.path.dirname(filename) + '/' + filename_base + '_' + suffix + '.' + filename_extension

        def GetValidFilename(filename: str, index: int):
            filename_new = AddFileSuffix(filename, str(index).zfill(5))

            if os.path.exists(filename_new):
                return GetValidFilename(filename, index + 1)

            return filename_new

        # set output filename with timestamp suffix
        output_filename = AddFileSuffix(data['output_path'], datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        if os.path.exists(output_filename):
            if data.get('existing_file_rule', "fail") == "fail":
                return {
                    'info': "File \"" + output_filename + "\" exists already. [existing_file_rule == fail]",
                    'error': "File exists"
                }
            elif data.get('existing_file_rule', "fail") == "increment":
                output_filename = GetValidFilename(output_filename, 0)

        try:
            wavfile.write(output_filename, resultPretty['sample_rate'], resultPretty['audio'])
        except PermissionError:
            return {
                'info': "Permission denied, couldn't create output file \"" + output_filename + '"',
                'error': "Permission denied"
            }

        resultPretty['file'] = output_filename

        # meta data is written by default, keep in mind that it will be lost after converting the file to another
        # format, if you want to keep the meta you have to take care of that yourself
        if data.get('meta_write_data', True):
            tags = {}
            dataCopy = {}
            dataCopy.update(data)

            # store source file tags
            if data.get('meta_copy_source_meta', True):
                with taglib.File(data['input_audio_path'], save_on_exit=False) as audio_file_source:
                    tags = audio_file_source.tags

            with taglib.File(output_filename, save_on_exit=True) as audio_file:
                # unnecessary flag for meta data
                if dataCopy.get('action'):
                    dataCopy.pop('action')

                # strip any path information by default
                if data.get('meta_strip_path', True):
                    dataCopy['output_path'] = os.path.basename(data['output_path'])
                    dataCopy['model_path'] = os.path.basename(data['model_path'])
                    dataCopy['input_audio_path'] = os.path.basename(data['input_audio_path'])

                audio_file.tags = tags
                audio_file.tags['rvc'] = json.dumps(dataCopy)



    return resultPretty