import os
import torch

# os.system("wget -P cvec/ https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt")
import gradio as gr
import librosa
import numpy as np
import logging
from fairseq import checkpoint_utils
from vc_infer_pipeline import VC
import traceback
from config import Config
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from i18n import I18nAuto

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

i18n = I18nAuto()
i18n.print()

config = Config()

weight_root = "weights"
weight_uvr5_root = "uvr5_weights"
index_root = "logs"
names = []
hubert_model = None
for name in os.listdir(weight_root):
    if name.endswith(".pth"):
        names.append(name)
index_paths = []
for root, dirs, files in os.walk(index_root, topdown=False):
    for name in files:
        if name.endswith(".index") and "trained" not in name:
            index_paths.append("%s/%s" % (root, name))


def get_vc(sid):
    global n_spk, tgt_sr, net_g, vc, cpt, version
    if sid == "" or sid == []:
        global hubert_model
        if hubert_model != None:  # 考虑到轮询, 需要加个判断看是否 sid 是由有模型切换到无模型的
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
            cpt = None
        return {"visible": False, "__type__": "update"}
    person = "%s/%s" % (weight_root, sid)
    print("loading %s" % person)
    cpt = torch.load(person, map_location="cpu")
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
    return {"visible": True, "maximum": n_spk, "__type__": "update"}


def load_hubert():
    global hubert_model
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


def vc_single(
    sid,
    input_audio_path,
    f0_up_key,
    f0_file,
    f0_method,
    file_index,
    file_index2,
    # file_big_npy,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect,
):  # spk_item, input_audio0, vc_transform0,f0_file,f0method0
    global tgt_sr, net_g, vc, hubert_model, version
    if input_audio_path is None:
        return "You need to upload an audio", None
    f0_up_key = int(f0_up_key)
    try:
        audio = input_audio_path[1] / 32768.0
        if len(audio.shape) == 2:
            audio = np.mean(audio, -1)
        audio = librosa.resample(audio, orig_sr=input_audio_path[0], target_sr=16000)
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max
        times = [0, 0, 0]
        if hubert_model == None:
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
        if resample_sr >= 16000 and tgt_sr != resample_sr:
            tgt_sr = resample_sr
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
        ), (tgt_sr, audio_opt)
    except:
        info = traceback.format_exc()
        print(info)
        return info, (None, None)


app = gr.Blocks()
with app:
    with gr.Tabs():
        with gr.TabItem("在线demo"):
            gr.Markdown(
                value="""
                RVC 在线demo
                """
            )
            sid = gr.Dropdown(label=i18n("推理音色"), choices=sorted(names))
            with gr.Column():
                spk_item = gr.Slider(
                    minimum=0,
                    maximum=2333,
                    step=1,
                    label=i18n("请选择说话人id"),
                    value=0,
                    visible=False,
                    interactive=True,
                )
            sid.change(
                fn=get_vc,
                inputs=[sid],
                outputs=[spk_item],
            )
            gr.Markdown(
                value=i18n("男转女推荐+12key, 女转男推荐-12key, 如果音域爆炸导致音色失真也可以自己调整到合适音域. ")
            )
            vc_input3 = gr.Audio(label="上传音频（长度小于90秒）")
            vc_transform0 = gr.Number(label=i18n("变调(整数, 半音数量, 升八度12降八度-12)"), value=0)
            f0method0 = gr.Radio(
                label=i18n("选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,crepe效果好但吃GPU"),
                choices=["pm", "harvest", "crepe", "rmvpe"],
                value="pm",
                interactive=True,
            )
            filter_radius0 = gr.Slider(
                minimum=0,
                maximum=7,
                label=i18n(">=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音"),
                value=3,
                step=1,
                interactive=True,
            )
            with gr.Column():
                file_index1 = gr.Textbox(
                    label=i18n("特征检索库文件路径,为空则使用下拉的选择结果"),
                    value="",
                    interactive=False,
                    visible=False,
                )
            file_index2 = gr.Dropdown(
                label=i18n("自动检测index路径,下拉式选择(dropdown)"),
                choices=sorted(index_paths),
                interactive=True,
            )
            index_rate1 = gr.Slider(
                minimum=0,
                maximum=1,
                label=i18n("检索特征占比"),
                value=0.88,
                interactive=True,
            )
            resample_sr0 = gr.Slider(
                minimum=0,
                maximum=48000,
                label=i18n("后处理重采样至最终采样率，0为不进行重采样"),
                value=0,
                step=1,
                interactive=True,
            )
            rms_mix_rate0 = gr.Slider(
                minimum=0,
                maximum=1,
                label=i18n("输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络"),
                value=1,
                interactive=True,
            )
            protect0 = gr.Slider(
                minimum=0,
                maximum=0.5,
                label=i18n("保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果"),
                value=0.33,
                step=0.01,
                interactive=True,
            )
            f0_file = gr.File(label=i18n("F0曲线文件, 可选, 一行一个音高, 代替默认F0及升降调"))
            but0 = gr.Button(i18n("转换"), variant="primary")
            vc_output1 = gr.Textbox(label=i18n("输出信息"))
            vc_output2 = gr.Audio(label=i18n("输出音频(右下角三个点,点了可以下载)"))
            but0.click(
                vc_single,
                [
                    spk_item,
                    vc_input3,
                    vc_transform0,
                    f0_file,
                    f0method0,
                    file_index1,
                    file_index2,
                    # file_big_npy1,
                    index_rate1,
                    filter_radius0,
                    resample_sr0,
                    rms_mix_rate0,
                    protect0,
                ],
                [vc_output1, vc_output2],
            )


app.launch()
