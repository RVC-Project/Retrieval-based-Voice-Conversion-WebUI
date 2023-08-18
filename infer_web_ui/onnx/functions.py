import gradio as gr


def load_onnx_tab(i18n, export_onnx):
    with gr.Row():
        ckpt_dir = gr.Textbox(label=i18n("RVC模型路径"), value="", interactive=True)
    with gr.Row():
        onnx_dir = gr.Textbox(
            label=i18n("Onnx输出路径"), value="", interactive=True
        )
    with gr.Row():
        info_onnx = gr.Label(label="info")
    with gr.Row():
        but_onnx = gr.Button(i18n("导出Onnx模型"), variant="primary")
    but_onnx.click(
        export_onnx, [ckpt_dir, onnx_dir], info_onnx, api_name="export_onnx"
    )
