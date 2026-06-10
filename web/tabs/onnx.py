"""ONNX export tab."""

import gradio as gr

from web.runtime import i18n
from web.train_ops import export_onnx


def build():
    with gr.TabItem(i18n("Onnx导出")):
        with gr.Row():
            ckpt_dir = gr.Textbox(label=i18n("RVC模型路径"), value="", interactive=True)
        with gr.Row():
            onnx_dir = gr.Textbox(
                label=i18n("Onnx输出路径"), value="", interactive=True
            )
        with gr.Row():
            infoOnnx = gr.Label(label="info")
        with gr.Row():
            butOnnx = gr.Button(i18n("导出Onnx模型"), variant="primary")
        butOnnx.click(
            export_onnx, [ckpt_dir, onnx_dir], infoOnnx, api_name="export_onnx"
        )
