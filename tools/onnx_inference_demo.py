import soundfile

from ..infer.lib.infer_pack.onnx_inference import OnnxRVC

hop_size = 512
sampling_rate = 40000  # Sample rate
f0_up_key = 0  # Pitch up/down
sid = 0  # Character ID
f0_method = "dio"  # F0 Extract algorithm
model_path = "ShirohaRVC.onnx"  # Full path of the model
vec_name = (
    "vec-256-layer-9"  # Internally padded to f"pretrained/{vec_name}.onnx", needs the vec model of onnx
)
wav_path = "123.wav"  # Input path or ByteIO instance
out_path = "out.wav"  # Output path or ByteIO instance

model = OnnxRVC(
    model_path, vec_path=vec_name, sr=sampling_rate, hop_size=hop_size, device="cuda"
)

audio = model.inference(wav_path, sid, f0_method=f0_method, f0_up_key=f0_up_key)

soundfile.write(out_path, audio, sampling_rate)
