import os

import boto3
import runpod
from dotenv import load_dotenv

load_dotenv()
from scipy.io import wavfile

from configs.config import Config
from infer.modules.vc.modules import VC


def process(job):
    job_input = job["input"]
    s3 = boto3.client('s3')
    bucket = job_input['bucket']  # voicerary
    s3_filepath = job_input['filepath']  # vc/audio/QcjsPLUasGO4C7TZnbRN2TSibl3XbBdWSUVI8Wzl.aif
    local_filepath = os.path.basename(s3_filepath)  # QcjsPLUasGO4C7TZnbRN2TSibl3XbBdWSUVI8Wzl.aif
    local_filepath_no_ext = os.path.splitext(local_filepath)[0]  # QcjsPLUasGO4C7TZnbRN2TSibl3XbBdWSUVI8Wzl
    s3_filepath_no_ext = os.path.splitext(s3_filepath)[0]  # vc/audio/QcjsPLUasGO4C7TZnbRN2TSibl3XbBdWSUVI8Wzl
    s3_converted_filepath = s3_filepath_no_ext + '_done.wav'  # vc/audio/QcjsPLUasGO4C7TZnbRN2TSibl3XbBdWSUVI8Wzl_done.wav
    index_root = os.getenv("index_root")
    index_path = f'{index_root}/{job_input["model_name"]}.index'

    try:
        s3.download_file(bucket, s3_filepath, local_filepath)

        config = Config()
        vc = VC(config)
        vc.get_vc(job_input["model_name"] + '.pth')
        _, wav_opt = vc.vc_single(
            0,
            local_filepath,
            job_input["f0up_key"],
            None,
            job_input["f0method"],
            index_path if os.path.isfile(index_path) else '',
            None,
            job_input["index_rate"],
            job_input["filter_radius"],
            job_input["resample_sr"],
            job_input["rms_mix_rate"],
            job_input["protect"],
        )
        wavfile.write(local_filepath_no_ext + '.wav', wav_opt[0], wav_opt[1])

        s3.upload_file(local_filepath_no_ext + '.wav', bucket, s3_converted_filepath)
    except Exception:
        return {"error": "Operation failed"}
    else:
        return {"converted": s3_converted_filepath}


runpod.serverless.start({"handler": process})
