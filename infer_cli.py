#!/usr/bin/env python

import os
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)

###
# USAGE
# In your Terminal or CMD or whatever
# python infer_cli.py [TRANSPOSE_VALUE] "[INPUT_PATH]" "[OUTPUT_PATH]" "[MODEL_PATH]" "[INDEX_FILE_PATH]" "[INFERENCE_DEVICE]" "[METHOD]"

device = "cuda:0"
is_half = False

if len(sys.argv) == 8:
    f0_up_key = int(sys.argv[1])  # transpose value
    input_path = sys.argv[2]
    output_path = sys.argv[3]
    model_path = sys.argv[4]
    file_index = sys.argv[5]  # .index file
    device = sys.argv[6]
    f0_method = sys.argv[7]  # pm or harvest or crepe

    # file_index2=sys.argv[8]
    # index_rate=float(sys.argv[10]) #search feature ratio
    # filter_radius=float(sys.argv[11]) #median filter
    # resample_sr=float(sys.argv[12]) #resample audio in post processing
    # rms_mix_rate=float(sys.argv[13]) #search feature
    print(sys.argv)
else:
    print('please provide the necessary arguments:')
    print('# infer_cli.py [TRANSPOSE_VALUE] "[INPUT_PATH]" "[OUTPUT_PATH]" "[MODEL_PATH]" "[INDEX_FILE_PATH]" "[INFERENCE_DEVICE]" "[METHOD]"')
    exit(-1)


# we need to wipe the arguments because the config module also tries to parse arguments and fails with the CLI arguments...
sys.argv = sys.argv[:1]
from src.configuration import config

config.device = device
config.is_half = is_half
config.device_config()


from vc_infer_utils import vc_single_json

result = vc_single_json({
        'sid': 0,
        'input_audio_path': input_path,
        'f0_up_key': f0_up_key,
        'f0_file': None,
        'f0_method': f0_method,
        'file_index': file_index,
        'file_index2': "",
        'index_rate': 1,
        'filter_radius': 3,
        'resample_sr': 0,
        'rms_mix_rate': 0,
        'model_path': model_path,
        'output_path': output_path
    })

print(result['info'])
