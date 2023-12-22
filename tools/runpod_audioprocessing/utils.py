import glob
import os
import tempfile

import librosa
import math
import numpy as np
import soundfile as sf
import torch
import statistics as stats
from dotenv import load_dotenv

from infer.lib.rmvpe import RMVPE

load_dotenv()


class AudioProcessingUtils:

    @staticmethod
    def get_temp_file_path(name: str) -> str:
        return os.path.join(tempfile.gettempdir(), name)

    @staticmethod
    def append_to_temp_file(temp_file_list: str, line: str) -> None:
        with open(temp_file_list, 'a') as f:
            f.write(line)

    @staticmethod
    def remove_temp_files(pattern: str) -> None:
        for temp_file in glob.glob(os.path.join(tempfile.gettempdir(), pattern)):
            os.remove(temp_file)

    @staticmethod
    def semitone_distance(pitch1: float, pitch2: float) -> int:
        if pitch1 == 0.0 or pitch2 == 0.0:
            return 0

        return round(12 * math.log2(pitch1 / pitch2))

    @staticmethod
    def get_fundamental_frequency(wav_audio_file_path: str) -> float:
        if os.path.isfile(wav_audio_file_path):
            audio, sampling_rate = sf.read(wav_audio_file_path)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio.transpose(1, 0))
            if sampling_rate != 16000:
                audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
            thred = 0.03  # 0.01
            device = "cuda" if torch.cuda.is_available() else "cpu"
            rmvpe = RMVPE(os.path.join(os.environ["RMVPE_ROOT"],
                                       'rmvpe.pt'), is_half=False, device=device)
            f0 = rmvpe.infer_from_audio(audio, thred=thred)
            f0_nonzero = f0[f0 != 0]
            # median_nonzero = np.median(f0_nonzero)
            # median = np.median(f0)
            mode_nonzero = stats.multimode(np.round(f0_nonzero))
            # mode = stats.multimode(np.round(f0))
            # print(f"F0 with 0s: median value is {round(median, 2)}Hz, mode value is {np.average(mode)}Hz")
            # print(f"F0 array without 0s: median value is {round(median_nonzero, 2)}Hz, mode value is {np.average(mode_nonzero)}Hz")
            return np.average(mode_nonzero)
        else:
            return 0
