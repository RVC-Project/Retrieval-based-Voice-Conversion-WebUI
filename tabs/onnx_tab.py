import gradio as gr
from shared import i18n


def export_onnx(ModelPath, ExportedPath):
    from infer.modules.onnx.export import export_onnx as eo

    eo(ModelPath, ExportedPath)


def create_onnx_tab():
    with gr.TabItem(i18n("Onnx Export")):
        with gr.Row():
            ckpt_dir = gr.Textbox(label=i18n("RVCModel path"), value="", interactive=True)
        with gr.Row():
            onnx_dir = gr.Textbox(
                label=i18n("OnnxOutput path"), value="", interactive=True
            )
        with gr.Row():
            infoOnnx = gr.Label(label="info")
        with gr.Row():
            butOnnx = gr.Button(i18n("Export Onnx model"), variant="primary")
        butOnnx.click(
            export_onnx, [ckpt_dir, onnx_dir], infoOnnx, api_name="export_onnx"
        )
