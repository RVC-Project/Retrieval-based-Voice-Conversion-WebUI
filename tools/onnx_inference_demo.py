import soundfile

from ..infer.lib.infer_pack.onnx_inference import OnnxRVC

hop_size = 512
sampling_rate = 40000  # 采样率
f0_up_key = 0  # 升降调
sid = 0  # 角色ID
f0_method = "dio"  # F0提取算法
model_path = "ShirohaRVC.onnx"  # 模型的完整路径
vec_name = (
    "vec-256-layer-9"  # 内部自动补齐为 f"pretrained/{vec_name}.onnx" 需要onnx的vec模型
)
wav_path = "123.wav"  # 输入路径或ByteIO实例
out_path = "out.wav"  # 输出路径或ByteIO实例

model = OnnxRVC(
    model_path, vec_path=vec_name, sr=sampling_rate, hop_size=hop_size, device="cuda"
)

audio = model.inference(wav_path, sid, f0_method=f0_method, f0_up_key=f0_up_key)

soundfile.write(out_path, audio, sampling_rate)
