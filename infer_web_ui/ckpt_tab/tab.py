import gradio as gr

from infer_web_ui.ckpt_tab.functions import change_info_
from infer_web_ui.utils import i18n
from lib.train.process_ckpt import change_info, extract_small_model, merge, show_info


def load_ckpt_tab():
    with gr.Group():
        gr.Markdown(value=i18n("模型融合, 可用于测试音色融合"))
        with gr.Row():
            ckpt_a = gr.Textbox(label=i18n("A模型路径"), value="", interactive=True)
            ckpt_b = gr.Textbox(label=i18n("B模型路径"), value="", interactive=True)
            alpha_a = gr.Slider(
                minimum=0,
                maximum=1,
                label=i18n("A模型权重"),
                value=0.5,
                interactive=True,
            )
        with gr.Row():
            sr_ = gr.Radio(
                label=i18n("目标采样率"),
                choices=["40k", "48k"],
                value="40k",
                interactive=True,
            )
            if_f0_ = gr.Radio(
                label=i18n("模型是否带音高指导"),
                choices=[i18n("是"), i18n("否")],
                value=i18n("是"),
                interactive=True,
            )
            info__ = gr.Textbox(
                label=i18n("要置入的模型信息"), value="", max_lines=8, interactive=True
            )
            name_to_save0 = gr.Textbox(
                label=i18n("保存的模型名不带后缀"),
                value="",
                max_lines=1,
                interactive=True,
            )
            version_2 = gr.Radio(
                label=i18n("模型版本型号"),
                choices=["v1", "v2"],
                value="v1",
                interactive=True,
            )
        with gr.Row():
            but6 = gr.Button(i18n("融合"), variant="primary")
            info4 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8)
        but6.click(
            merge,
            [
                ckpt_a,
                ckpt_b,
                alpha_a,
                sr_,
                if_f0_,
                info__,
                name_to_save0,
                version_2,
            ],
            info4,
            api_name="ckpt_merge",
        )  # def merge(path1,path2,alpha1,sr,f0,info):
    with gr.Group():
        gr.Markdown(value=i18n("修改模型信息(仅支持weights文件夹下提取的小模型文件)"))
        with gr.Row():
            ckpt_path0 = gr.Textbox(
                label=i18n("模型路径"), value="", interactive=True
            )
            info_ = gr.Textbox(
                label=i18n("要改的模型信息"), value="", max_lines=8, interactive=True
            )
            name_to_save1 = gr.Textbox(
                label=i18n("保存的文件名, 默认空为和源文件同名"),
                value="",
                max_lines=8,
                interactive=True,
            )
        with gr.Row():
            but7 = gr.Button(i18n("修改"), variant="primary")
            info5 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8)
        but7.click(
            change_info,
            [ckpt_path0, info_, name_to_save1],
            info5,
            api_name="ckpt_modify",
        )
    with gr.Group():
        gr.Markdown(value=i18n("查看模型信息(仅支持weights文件夹下提取的小模型文件)"))
        with gr.Row():
            ckpt_path1 = gr.Textbox(
                label=i18n("模型路径"), value="", interactive=True
            )
            but8 = gr.Button(i18n("查看"), variant="primary")
            info6 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8)
        but8.click(show_info, [ckpt_path1], info6, api_name="ckpt_show")
    with gr.Group():
        gr.Markdown(
            value=i18n(
                "模型提取(输入logs文件夹下大文件模型路径),适用于训一半不想训了模型没有自动提取保存小文件模型,或者想测试中间模型的情况"
            )
        )
        with gr.Row():
            ckpt_path2 = gr.Textbox(
                label=i18n("模型路径"),
                value="E:\\codes\\py39\\logs\\mi-test_f0_48k\\G_23333.pth",
                interactive=True,
            )
            save_name = gr.Textbox(
                label=i18n("保存名"), value="", interactive=True
            )
            sr__ = gr.Radio(
                label=i18n("目标采样率"),
                choices=["32k", "40k", "48k"],
                value="40k",
                interactive=True,
            )
            if_f0__ = gr.Radio(
                label=i18n("模型是否带音高指导,1是0否"),
                choices=["1", "0"],
                value="1",
                interactive=True,
            )
            version_1 = gr.Radio(
                label=i18n("模型版本型号"),
                choices=["v1", "v2"],
                value="v2",
                interactive=True,
            )
            info___ = gr.Textbox(
                label=i18n("要置入的模型信息"), value="", max_lines=8, interactive=True
            )
            but9 = gr.Button(i18n("提取"), variant="primary")
            info7 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8)
            ckpt_path2.change(
                change_info_, [ckpt_path2], [sr__, if_f0__, version_1]
            )
        but9.click(
            extract_small_model,
            [ckpt_path2, save_name, sr__, if_f0__, info___, version_1],
            info7,
            api_name="ckpt_extract",
        )
