
import torch
from infer.modules.vc.standalone_infer import VC
from configs.config import Config
import numpy as np
import librosa
import soundfile as sf

# Model setup
device = 'cpu:0' # 'cuda:0' for GPU
is_half = False # True for half precision
model_path = '<model-path>.pth' 
index_path = '<index-path>.index'
config = Config(device=device,is_half=is_half)
config.hubert_model_path = 'assets/hubert/hubert_base.pt'

# Conversion parameters
f0up_key = -6 
f0method = 'crepe'
index_rate = 0.6 
filter_radius = 2
resample_sr = 0
rms_mix_rate = 0.5
protect=0

# Input audio
audio, sr = librosa.load('<audio-path>')
input_audio = librosa.resample(audio, sr, 16000)

# Instantiate model
vc = VC(config)
vc.get_vc(model_path)

# Inference
audio_opt = vc.vc_single(
                0,
                input_audio,
                'output.wav',
                f0up_key,
                None,
                f0method,
                index_path,
                None,
                index_rate,
                filter_radius,
                resample_sr,
                rms_mix_rate,
                protect,
            )

# Save output audio
message, audio_info = audio_opt
sr, output_audio = audio_info
sf.write('output.wav', output_audio, sr)