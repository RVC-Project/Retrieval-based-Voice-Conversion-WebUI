import os
import traceback

import librosa
import numpy as np
import av
from io import BytesIO


def wav2(i, o, format):
    inp = av.open(i, "rb")
    if format == "m4a":
        format = "mp4"
    out = av.open(o, "wb", format=format)
    if format == "ogg":
        format = "libvorbis"
    if format == "mp4":
        format = "aac"

    ostream = out.add_stream(format)

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)

    for p in ostream.encode(None):
        out.mux(p)

    out.close()
    inp.close()


def audio2(i, o, format, sr):
    inp = av.open(i, "rb")
    out = av.open(o, "wb", format=format)
    if format == "ogg":
        format = "libvorbis"
    if format == "f32le":
        format = "pcm_f32le"

    ostream = out.add_stream(format, channels=1)
    ostream.sample_rate = sr

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)

    out.close()
    inp.close()


def load_audio(file: str | bytes, sr: int):
    # open audio file
    if type(file) == bytes:
        opened_file = BytesIO(file)
    else:
        file = (
            file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )  # 防止小白拷路径头尾带了空格和"和回车
        if os.path.exists(file) == False:
            raise RuntimeError(
                "You input a wrong audio path that does not exists, please fix it!"
            )
        opened_file = open(file, "rb")

    # convert audio file to 16k wav
    try:
        with BytesIO() as out:
            audio2(opened_file, out, "f32le", sr)
            ret = np.frombuffer(out.getvalue(), np.float32).flatten()
    except AttributeError:
        audio = file[1] / 32768.0
        if len(audio.shape) == 2:
            audio = np.mean(audio, -1)
        ret = librosa.resample(audio, orig_sr=file[0], target_sr=16000)

    except:
        raise RuntimeError(traceback.format_exc())

    # close audio file to be safe
    opened_file.close()

    return ret
