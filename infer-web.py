import logging
import os
import shutil
import sys
import warnings

import gradio as gr
import torch

from infer_web_ui.ckpt_tab.tab import load_ckpt_tab
from infer_web_ui.faq_tab.tab import load_faq
from infer_web_ui.inference_tab.tab import load_inference_tab
from infer_web_ui.onnx.tab import load_onnx_tab
from infer_web_ui.train_tab.tab import load_train_tab

sys.path.append(os.getcwd())

from infer_web_ui.separation_tab.tab import load_separation_tab
from infer_web_ui.utils import i18n, now_dir, tmp, config

# Get logger
logging.getLogger("numba").setLevel(logging.WARNING)

# Manage paths
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/infer_pack" % now_dir, ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/uvr5_pack" % now_dir, ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "weights"), exist_ok=True)

# Set random seed
warnings.filterwarnings("ignore")
torch.manual_seed(114514)

# Environment
os.environ["TEMP"] = tmp
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"


# Gradio UI
with gr.Blocks(title="RVC WebUI") as app:
    # Presentation tab
    gr.Markdown(
        value="## RVC WebUI\n"
    )

    # Tabs
    with gr.Tabs():

        # Inference tab
        with gr.TabItem(i18n("模型推理")):
            load_inference_tab()

        # Separation & Reverb Removal tab
        with gr.TabItem(i18n("伴奏人声分离&去混响&去回声")):
            load_separation_tab()

        # Train tab
        with gr.TabItem(i18n("训练")):
            load_train_tab()
        # ckpt tab
        with gr.TabItem(i18n("ckpt处理")):
            load_ckpt_tab()

        # Onnx tab
        with gr.TabItem(i18n("Onnx导出")):
            load_onnx_tab()

        # Faq tab
        tab_faq = i18n("常见问题解答")
        with gr.TabItem(tab_faq):
            load_faq(tab_faq)

    # License
    gr.Markdown(
        value=i18n(
            "本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>LICENSE</b>."
        )
    )

    if config.iscolab:
        app.queue(concurrency_count=511, max_size=1022).launch(share=True)
    else:
        app.queue(concurrency_count=511, max_size=1022).launch(
            server_name="0.0.0.0",
            inbrowser=not config.noautoopen,
            server_port=config.listen_port,
            quiet=True,
        )
