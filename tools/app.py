import logging
import os

# os.system("wget -P cvec/ https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt")
import gradio as gr
from dotenv import load_dotenv

from configs.config import Config
from i18n.i18n import I18nAuto
from infer.modules.vc.modules import VC

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

i18n = I18nAuto()
logger.info(i18n)

load_dotenv()
config = Config()
vc = VC(config)

weight_root = os.getenv("weight_root")
weight_uvr5_root = os.getenv("weight_uvr5_root")
index_root = os.getenv("index_root")
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
            sid.change(fn=vc.get_vc, inputs=[sid], outputs=[spk_item])
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
                vc.vc_single,
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
