#!/usr/bin/env python3
import os
import sys

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import faiss
faiss.omp_set_num_threads(1)

now_dir = os.getcwd()
sys.path.append(now_dir)

from dotenv import load_dotenv
load_dotenv()

os.environ['weight_root'] = 'assets/weights'
os.environ['index_root'] = 'logs'
os.environ['rmvpe_root'] = 'assets/rmvpe'

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")

from configs.config import Config
config = Config()
print(f"Device selected by config: {config.device}")

from infer.modules.vc.modules import VC
vc_instance = VC(config)

model_name = "Voice_New.pth"
input_audio = "/Users/arunkumarv/Music/Voice Clone/Voice_convert.mp3"
output_audio = "/Users/arunkumarv/Music/Voice Clone/rvc-webui/output/Voice_New/converted.wav"

os.makedirs(os.path.dirname(output_audio), exist_ok=True)

print(f"\nLoading model: {model_name}")
vc_instance.get_vc(model_name)

print(f"Converting audio: {input_audio}")
print(f"Output will be saved to: {output_audio}")

print("Starting vc_single...")
sys.stdout.flush()

import soundfile as sf

try:
    result_message, audio_result = vc_instance.vc_single(
        sid=0,
        input_audio_path=input_audio,
        f0_up_key=0,
        f0_file=None,
        f0_method="rmvpe",
        file_index=f"logs/Voice_New/added_IVF86_Flat_nprobe_1.index",
        file_index2="",
        index_rate=0.75,
        filter_radius=3,
        resample_sr=0,
        rms_mix_rate=0.25,
        protect=0.33
    )
    
    print(f"\nResult: {result_message}")
    
    sample_rate, audio_data = audio_result
    if audio_data is not None and sample_rate is not None:
        sf.write(output_audio, audio_data, sample_rate)
        print(f"✓ Audio saved successfully to: {output_audio}")
    else:
        print("✗ Conversion failed!")
        sys.exit(1)
except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()
    sys.exit(1)
