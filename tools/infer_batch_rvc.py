import os
import sys

print("Command-line arguments:", sys.argv)

now_dir = os.getcwd()
sys.path.append(now_dir)

import tqdm as tq
from dotenv import load_dotenv
from scipy.io import wavfile
from tap import Tap

from lib.f0 import PitchMethod


class InferBatchArgs(Tap):
    # Pitch shift in semitones.
    f0up_key: int = 0
    # Input directory containing wav files.
    input_path: str
    # Optional retrieval index path.
    index_path: str | None = None
    # F0 extraction method.
    f0method: PitchMethod = "harvest"
    # Output directory.
    opt_path: str
    # Model name stored in assets/weights.
    model_name: str
    # Retrieval index influence.
    index_rate: float = 0.66
    # Override inference device.
    device: str | None = None
    # Override half precision mode.
    is_half: bool | None = None
    # Median filter radius for extracted pitch.
    filter_radius: int = 3
    # Resample output sample rate, or 0 to keep model rate.
    resample_sr: int = 0
    # RMS envelope mix rate.
    rms_mix_rate: float = 1
    # Protect unvoiced consonants.
    protect: float = 0.33


def arg_parse() -> InferBatchArgs:
    args = InferBatchArgs().parse_args()
    sys.argv = sys.argv[:1]
    return args


def main() -> None:
    load_dotenv()
    args = arg_parse()
    from configs.config import Config
    from infer.lib.audio import load_audio
    from infer.modules.vc.modules import VC

    config = Config()
    config.device = args.device if args.device else config.device
    config.is_half = args.is_half if args.is_half is not None else config.is_half
    vc = VC(config)
    vc.get_vc(args.model_name)
    audios = os.listdir(args.input_path)
    for file in tq.tqdm(audios):
        if file.endswith(".wav"):
            file_path = os.path.join(args.input_path, file)
            audio = load_audio(file_path, 16000)
            message, wav_opt = vc.vc_single(
                (16000, audio),
                args.f0up_key,
                args.f0method,
                args.index_path,
                args.index_rate,
                args.resample_sr,
                args.rms_mix_rate,
                args.protect,
            )
            if wav_opt is None:
                raise RuntimeError(message)
            out_path = os.path.join(args.opt_path, file)
            wavfile.write(out_path, wav_opt[0], wav_opt[1])


if __name__ == "__main__":
    main()
