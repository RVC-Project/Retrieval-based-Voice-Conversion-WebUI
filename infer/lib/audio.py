import platform
import numpy as np
import av
import av.audio.resampler
import re
from av.audio.frame import AudioFrame
from av.audio.stream import AudioStream
from numpy.typing import NDArray


def wav2(input_path: str, output_path: str, format: str) -> None:
    inp = av.open(input_path, "r")
    if format == "m4a":
        format = "mp4"
    out = av.open(output_path, "w", format=format)
    if format == "ogg":
        format = "libvorbis"
    if format == "mp4":
        format = "aac"

    ostream = out.add_stream(format)
    if not isinstance(ostream, AudioStream):
        raise RuntimeError(f"Expected an audio stream for format {format}")

    for frame in inp.decode(audio=0):
        if not isinstance(frame, AudioFrame):
            continue
        for p in ostream.encode(frame):
            out.mux(p)

    for p in ostream.encode(None):
        out.mux(p)

    out.close()
    inp.close()


def load_audio(file: str, sr: int) -> NDArray[np.float32]:
    try:
        with av.open(file, "r") as container:
            stream = next(s for s in container.streams if s.type == "audio")

            resampler = av.audio.resampler.AudioResampler(
                format="flt", layout="mono", rate=sr
            )

            audio_data: list[NDArray[np.float32]] = []
            for frame in container.decode(stream):
                if not isinstance(frame, AudioFrame):
                    continue
                # Resample returns either a frame or a list of frames
                resampled = resampler.resample(frame)
                if not resampled:
                    continue
                if isinstance(resampled, list):
                    frames = resampled
                else:
                    frames = [resampled]

                for f in frames:
                    arr = np.asarray(f.to_ndarray(), dtype=np.float32)
                    audio_data.append(arr)

            return np.concatenate(audio_data, axis=1).flatten()
    except Exception as e:
        raise RuntimeError(f"Failed to load audio with PyAV: {e}") from e


def clean_path(path_str: str) -> str:
    if platform.system() == "Windows":
        path_str = path_str.replace("/", "\\")
    path_str = re.sub(
        r"[\u202a\u202b\u202c\u202d\u202e]", "", path_str
    )  # Remove Unicode control characters
    return path_str.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
