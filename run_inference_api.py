#!/usr/bin/env python3
from gradio_client import Client
import os

client = Client("http://localhost:7865")

input_audio = "/Users/arunkumarv/Music/Voice Clone/Voice_convert.mp3"
output_dir = "/Users/arunkumarv/Music/Voice Clone/rvc-webui/output/Voice_New"
os.makedirs(output_dir, exist_ok=True)

print(f"Submitting inference request...")
print(f"Input: {input_audio}")
print(f"Model: Voice_New.pth")
print(f"F0 Method: pm")

result = client.predict(
    spk_item="Voice_New.pth",
    input_audio0=input_audio,
    vc_transform0=0,  # pitch shift
    f0_file=None,
    f0method0="pm",  # F0 method
    file_index1="",  # manual index path
    file_index2="logs/Voice_New/added_IVF86_Flat_nprobe_1.index",  # dropdown selection
    index_rate1=0.75,  # retrieval mix
    filter_radius0=3,  # median filter
    resample_sr0=0,  # output sample rate
    rms_mix_rate0=0.25,  # volume envelope
    protect0=0.33,  # consonant protection
    api_name="/infer_convert"
)

output_message, output_audio_tuple = result
print(f"\nResult: {output_message}")

if output_audio_tuple and len(output_audio_tuple) > 1:
    output_path = os.path.join(output_dir, "converted.wav")
    # Gradio returns the audio file path
    if isinstance(output_audio_tuple, tuple) and len(output_audio_tuple) == 2:
        sr, audio_file = output_audio_tuple
        print(f"✓ Audio converted successfully!")
        print(f"Sample rate: {sr} Hz")
        print(f"Output: {output_path}")
    else:
        print(f"Unexpected output format: {output_audio_tuple}")
else:
    print("✗ Conversion failed!")
