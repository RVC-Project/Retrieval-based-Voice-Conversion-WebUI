import os
import traceback
from pathlib import Path
import gradio as gr
from shared import i18n
from infer.lib.train.process_ckpt import (
    change_info,
    extract_small_model,
    merge,
    show_info,
)


#                    ckpt_path2.change(change_info_,[ckpt_path2],[sr__,if_f0__])
def change_info_(ckpt_path):
    train_log = Path(ckpt_path).parent / "train.log"
    if not train_log.exists():
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}
    try:
        with open(train_log, "r") as f:
            info = eval(f.read().strip("\n").split("\n")[0].split("\t")[-1])
            sr, f0 = info["sample_rate"], info["if_f0"]
            version = "v2" if ("version" in info and info["version"] == "v2") else "v1"
            return sr, str(f0), version
    except:
        traceback.print_exc()
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}


def create_ckpt_processing_tab():
    with gr.TabItem(i18n("ckpt processing")):
        with gr.Group():
            gr.Markdown(value=i18n("Model fusion, can be used to test timbre fusion"))
            with gr.Row():
                ckpt_a = gr.Textbox(label=i18n("AModel path"), value="", interactive=True)
                ckpt_b = gr.Textbox(label=i18n("BModel path"), value="", interactive=True)
                alpha_a = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Model A weight"),
                    value=0.5,
                    interactive=True,
                )
            with gr.Row():
                sr_ = gr.Radio(
                    label=i18n("Target sample rate"),
                    choices=["40k", "48k"],
                    value="40k",
                    interactive=True,
                )
                if_f0_ = gr.Radio(
                    label=i18n("Does the model have pitch guidance?"),
                    choices=[i18n("Yes"), i18n("No")],
                    value=i18n("Yes"),
                    interactive=True,
                )
                info__ = gr.Textbox(
                    label=i18n("Model info to insert"),
                    value="",
                    max_lines=8,
                    interactive=True,
                )
                name_to_save0 = gr.Textbox(
                    label=i18n("Saved model name without extension"),
                    value="",
                    max_lines=1,
                    interactive=True,
                )
                version_2 = gr.Radio(
                    label=i18n("Model version type"),
                    choices=["v1", "v2"],
                    value="v1",
                    interactive=True,
                )
            with gr.Row():
                but6 = gr.Button(i18n("Fuse"), variant="primary")
                info4 = gr.Textbox(label=i18n("Output info"), value="", max_lines=8)
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
            gr.Markdown(
                value=i18n("Modify model info (only supports small model files extracted under the weights folder)")
            )
            with gr.Row():
                ckpt_path0 = gr.Textbox(
                    label=i18n("Model path"), value="", interactive=True
                )
                info_ = gr.Textbox(
                    label=i18n("Model info to modify"),
                    value="",
                    max_lines=8,
                    interactive=True,
                )
                name_to_save1 = gr.Textbox(
                    label=i18n("Saved filename, empty defaults to the same name as the source file"),
                    value="",
                    max_lines=8,
                    interactive=True,
                )
            with gr.Row():
                but7 = gr.Button(i18n("Modify"), variant="primary")
                info5 = gr.Textbox(label=i18n("Output info"), value="", max_lines=8)
            but7.click(
                change_info,
                [ckpt_path0, info_, name_to_save1],
                info5,
                api_name="ckpt_modify",
            )
        with gr.Group():
            gr.Markdown(
                value=i18n("View model info (only supports small model files extracted under the weights folder)")
            )
            with gr.Row():
                ckpt_path1 = gr.Textbox(
                    label=i18n("Model path"), value="", interactive=True
                )
                but8 = gr.Button(i18n("View"), variant="primary")
                info6 = gr.Textbox(label=i18n("Output info"), value="", max_lines=8)
            but8.click(show_info, [ckpt_path1], info6, api_name="ckpt_show")
        with gr.Group():
            gr.Markdown(
                value=i18n(
                    "Model Extract (input large Model path under logs folder), useful when you stop training halfway and the small model wasn't automatically saved, or for testing intermediate models"
                )
            )
            with gr.Row():
                ckpt_path2 = gr.Textbox(
                    label=i18n("Model path"),
                    value="E:\\codes\\py39\\logs\\mi-test_f0_48k\\G_23333.pth",
                    interactive=True,
                )
                save_name = gr.Textbox(label=i18n("Save name"), value="", interactive=True)
                sr__ = gr.Radio(
                    label=i18n("Target sample rate"),
                    choices=["32k", "40k", "48k"],
                    value="40k",
                    interactive=True,
                )
                if_f0__ = gr.Radio(
                    label=i18n("Does the model have pitch guidance? 1 for yes, 0 for no"),
                    choices=["1", "0"],
                    value="1",
                    interactive=True,
                )
                version_1 = gr.Radio(
                    label=i18n("Model version type"),
                    choices=["v1", "v2"],
                    value="v2",
                    interactive=True,
                )
                info___ = gr.Textbox(
                    label=i18n("Model info to insert"),
                    value="",
                    max_lines=8,
                    interactive=True,
                )
                but9 = gr.Button(i18n("Extract"), variant="primary")
                info7 = gr.Textbox(label=i18n("Output info"), value="", max_lines=8)
                ckpt_path2.change(
                    change_info_, [ckpt_path2], [sr__, if_f0__, version_1]
                )
            but9.click(
                extract_small_model,
                [ckpt_path2, save_name, sr__, if_f0__, info___, version_1],
                info7,
                api_name="ckpt_extract",
            )
