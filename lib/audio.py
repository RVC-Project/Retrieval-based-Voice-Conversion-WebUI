import ffmpeg
import numpy as np


def load_audio(file, sr):
    out=None
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        # 防止小白拷路径头尾带了空格和"和回车
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        raise RuntimeError(f"来自ffmpeg的异常(Exception from ffmpeg):\n\t{e}")

    return np.frombuffer(out, np.float32).flatten()
