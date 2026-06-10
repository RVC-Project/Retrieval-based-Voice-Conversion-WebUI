"""Assembles the full WebUI from the tab modules."""

import gradio as gr

from web.runtime import i18n
from web.tabs import ckpt, faq, inference, onnx, train, uvr5


def build_app() -> gr.Blocks:
    with gr.Blocks(title="RVC WebUI") as app:
        gr.Markdown("## RVC WebUI")
        gr.Markdown(
            value=i18n(
                "本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>LICENSE</b>."
            )
        )
        with gr.Tabs():
            inference.build()
            uvr5.build()
            train.build()
            ckpt.build()
            onnx.build()
            faq.build()
    return app
