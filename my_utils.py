import ffmpeg
import numpy as np
import re


def load_audio(file, sr):
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = (
            file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )  # 防止小白拷路径头尾带了空格和"和回车
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

    return np.frombuffer(out, np.float32).flatten()


def rename_as_wav_extension(file_name):
    '''
    Renames file_name extension as .wav
    * If a file has an extension different from `.wav` it will be renamed to `.wav`
      For example,file `abc.mp3` will be renamed as `abc.wav`
    * If a file does not have an extension, `.wav` will be appended as extension
      For example,file `abc` will be renamed as `abc.wav`
    `
    '''
    if '.' not in file_name:
        return file_name + '.wav'
    return re.sub(r'\.\w*?$', '.wav', file_name)
