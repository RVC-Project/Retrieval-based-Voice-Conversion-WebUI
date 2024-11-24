import argparse
from infer.modules.onnx.export import export_onnx as eo

# Usage Example:
# python tools/export_onnx.py --input_model ~/models/my-model.pth --output_model ~/models/my-model.onnx
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", type=str, help="input model path", default="Shiroha/shiroha.pth")
    parser.add_argument("--output_model", type=str, help="output Onnx model path", default="model.onnx")
    args = parser.parse_args()

    ModelPath = args.input_model
    ExportedPath = args.output_model
    eo(ModelPath, ExportedPath)
