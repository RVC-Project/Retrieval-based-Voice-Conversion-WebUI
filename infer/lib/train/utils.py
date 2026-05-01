import logging
import json
import os
import subprocess
import sys
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.io.wavfile import read
from tap import Tap

# MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging


class TrainArgs(Tap):
    # Checkpoint save frequency in epochs.
    save_every_epoch: int
    # Total training epochs.
    total_epoch: int
    # Pretrained generator path.
    pretrainG: str = ""
    # Pretrained discriminator path.
    pretrainD: str = ""
    # GPU IDs split by hyphen.
    gpus: str = "0"
    # Training batch size.
    batch_size: int
    # Experiment directory name under logs.
    experiment_dir: str
    # Sample rate, such as 32k, 40k, or 48k.
    sample_rate: str
    # Save extracted model weights when saving checkpoints.
    save_every_weights: str = "0"
    # Model version.
    version: str
    # Whether to use f0 as an input, 1 or 0.
    if_f0: int
    # Whether to save only the latest G/D pth files, 1 or 0.
    if_latest: int
    # Whether to cache the dataset in GPU memory, 1 or 0.
    if_cache_data_in_gpu: int

    def configure(self) -> None:
        self.add_argument("-se", "--save_every_epoch")
        self.add_argument("-te", "--total_epoch")
        self.add_argument("-pg", "--pretrainG")
        self.add_argument("-pd", "--pretrainD")
        self.add_argument("-g", "--gpus")
        self.add_argument("-bs", "--batch_size")
        self.add_argument("-e", "--experiment_dir")
        self.add_argument("-sr", "--sample_rate")
        self.add_argument("-sw", "--save_every_weights")
        self.add_argument("-v", "--version")
        self.add_argument("-f0", "--if_f0")
        self.add_argument("-l", "--if_latest")
        self.add_argument("-c", "--if_cache_data_in_gpu")


def load_checkpoint_d(
    checkpoint_path: Path, combd, sbd, optimizer=None, load_opt: int = 1
):
    assert checkpoint_path.is_file()
    checkpoint_dict = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )

    ##################
    def go(model, bkey):
        saved_state_dict = checkpoint_dict[bkey]
        if hasattr(model, "module"):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():  # Shape required by the model
            try:
                new_state_dict[k] = saved_state_dict[k]
                if saved_state_dict[k].shape != state_dict[k].shape:
                    logger.warning(
                        "shape-%s-mismatch. need: %s, get: %s",
                        k,
                        state_dict[k].shape,
                        saved_state_dict[k].shape,
                    )  #
                    raise KeyError
            except:
                # logger.info(traceback.format_exc())
                logger.info("%s is not in the checkpoint", k)  # Missing in pretrain
                new_state_dict[k] = v  # Random values provided by the model
        if hasattr(model, "module"):
            model.module.load_state_dict(new_state_dict, strict=False)
        else:
            model.load_state_dict(new_state_dict, strict=False)
        return model

    go(combd, "combd")
    model = go(sbd, "sbd")
    #############
    logger.info("Loaded model weights")

    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    if (
        optimizer is not None and load_opt == 1
    ):  ### If it cannot load, and it's empty, reinitialize it. It might also affect the update of the lr schedule, so catch it in the outermost layer of the train file
        #   try:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    #   except:
    #     traceback.print_exc()
    logger.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


# def load_checkpoint(checkpoint_path, model, optimizer=None):
#   assert os.path.isfile(checkpoint_path)
#   checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
#   iteration = checkpoint_dict['iteration']
#   learning_rate = checkpoint_dict['learning_rate']
#   if optimizer is not None:
#     optimizer.load_state_dict(checkpoint_dict['optimizer'])
#   # print(1111)
#   saved_state_dict = checkpoint_dict['model']
#   # print(1111)
#
#   if hasattr(model, 'module'):
#     state_dict = model.module.state_dict()
#   else:
#     state_dict = model.state_dict()
#   new_state_dict= {}
#   for k, v in state_dict.items():
#     try:
#       new_state_dict[k] = saved_state_dict[k]
#     except:
#       logger.info("%s is not in the checkpoint" % k)
#       new_state_dict[k] = v
#   if hasattr(model, 'module'):
#     model.module.load_state_dict(new_state_dict)
#   else:
#     model.load_state_dict(new_state_dict)
#   logger.info("Loaded checkpoint '{}' (epoch {})" .format(
#     checkpoint_path, iteration))
#   return model, optimizer, learning_rate, iteration
def load_checkpoint(checkpoint_path: Path, model, optimizer=None, load_opt: int = 1):
    assert checkpoint_path.is_file()
    checkpoint_dict = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )

    saved_state_dict = checkpoint_dict["model"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():  # Shape required by the model
        try:
            new_state_dict[k] = saved_state_dict[k]
            if saved_state_dict[k].shape != state_dict[k].shape:
                logger.warning(
                    "shape-%s-mismatch|need-%s|get-%s",
                    k,
                    state_dict[k].shape,
                    saved_state_dict[k].shape,
                )  #
                raise KeyError
        except:
            # logger.info(traceback.format_exc())
            logger.info("%s is not in the checkpoint", k)  # Missing in pretrain
            new_state_dict[k] = v  # Random values provided by the model
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)
    logger.info("Loaded model weights")

    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    if (
        optimizer is not None and load_opt == 1
    ):  ### If it cannot load, and it's empty, reinitialize it. It might also affect the update of the lr schedule, so catch it in the outermost layer of the train file
        #   try:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    #   except:
    #     traceback.print_exc()
    logger.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path: Path):
    logger.info(
        "Saving model and optimizer state at epoch {} to {}".format(
            iteration, checkpoint_path
        )
    )
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(
        {
            "model": state_dict,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


def save_checkpoint_d(
    combd, sbd, optimizer, learning_rate: float, iteration, checkpoint_path: Path
):
    logger.info(
        "Saving model and optimizer state at epoch {} to {}".format(
            iteration, checkpoint_path
        )
    )
    if hasattr(combd, "module"):
        state_dict_combd = combd.module.state_dict()
    else:
        state_dict_combd = combd.state_dict()
    if hasattr(sbd, "module"):
        state_dict_sbd = sbd.module.state_dict()
    else:
        state_dict_sbd = sbd.state_dict()
    torch.save(
        {
            "combd": state_dict_combd,
            "sbd": state_dict_sbd,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


class SummaryWriter(Protocol):
    def add_scalar(self, tag: str, scalar_value: object, global_step: int) -> object: ...

    def add_histogram(self, tag: str, values: object, global_step: int) -> object: ...

    def add_image(
        self, tag: str, img_tensor: NDArray[np.generic], global_step: int, dataformats: str
    ) -> object: ...

    def add_audio(
        self, tag: str, snd_tensor: NDArray[np.generic], global_step: int, sample_rate: int
    ) -> object: ...


def summarize(
    writer: SummaryWriter,
    global_step: int,
    scalars: Mapping[str, object] | None = None,
    histograms: Mapping[str, object] | None = None,
    images: Mapping[str, NDArray[np.generic]] | None = None,
    audios: Mapping[str, NDArray[np.generic]] | None = None,
    audio_sampling_rate: int = 22050,
) -> None:
    scalars = scalars or {}
    histograms = histograms or {}
    images = images or {}
    audios = audios or {}
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


def latest_checkpoint_path(dir_path: Path, regex: str = "G_*.pth") -> Path:
    f_list = sorted(dir_path.glob(regex), key=lambda f: int("".join(filter(str.isdigit, f.name))))
    x = f_list[-1]
    logger.debug(x)
    return x


def load_wav_to_torch(full_path: Path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename: Path, split="|"):
    try:
        with open(filename, encoding="utf-8") as f:
            filepaths_and_text = [line.strip().split(split) for line in f]
    except UnicodeDecodeError:
        with open(filename) as f:
            filepaths_and_text = [line.strip().split(split) for line in f]

    return filepaths_and_text


def get_hparams(init=True):
    """
    todo:
      The ending group of seven:
        Save frequency, total epochs                    done
        bs                                    done
        pretrainG、pretrainD                  done
        Card number: os.environ["CUDA_VISIBLE_DEVICES"]   done
        if_latest                             done
      Model: if_f0                             done
      Sample rate: Auto-select config                  done
      Whether to cache dataset into GPU: if_cache_data_in_gpu done

      -m:
        Auto-determine training_files path, change hps.data.training_files in train_nsf_load_pretrain.py    done
      -c is no longer needed
    """
    args = TrainArgs().parse_args()
    name = args.experiment_dir
    experiment_dir = Path("./logs") / args.experiment_dir

    config_save_path = experiment_dir / "config.json"
    with open(config_save_path, "r") as f:
        config = json.load(f)

    hparams = HParams(**config)
    hparams.model_dir = hparams.experiment_dir = str(experiment_dir)
    hparams.save_every_epoch = args.save_every_epoch
    hparams.name = name
    hparams.total_epoch = args.total_epoch
    hparams.pretrainG = args.pretrainG
    hparams.pretrainD = args.pretrainD
    hparams.version = args.version
    hparams.gpus = args.gpus
    hparams.train.batch_size = args.batch_size
    hparams.sample_rate = args.sample_rate
    hparams.if_f0 = args.if_f0
    hparams.if_latest = args.if_latest
    hparams.save_every_weights = args.save_every_weights
    hparams.if_cache_data_in_gpu = args.if_cache_data_in_gpu
    hparams.data.training_files = str(experiment_dir / "filelist.txt")
    return hparams


def get_hparams_from_dir(model_dir: Path):
    config_save_path = model_dir / "config.json"
    with open(config_save_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    hparams.model_dir = str(model_dir)
    return hparams


def get_hparams_from_file(config_path: Path):
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams


def check_git_hash(model_dir: str):
    source_dir = Path(os.path.realpath(__file__)).parent
    git_check = subprocess.run(
        ["git", "-C", str(source_dir), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    if git_check.returncode != 0:
        return

    cur_hash = subprocess.check_output(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        text=True,
    ).strip()

    git_hash_file = Path(model_dir) / "githash"
    if git_hash_file.exists():
        saved_hash = git_hash_file.read_text()
        if saved_hash != cur_hash:
            logger.warning(
                "git hash values are different. {}(saved) != {}(current)".format(
                    saved_hash[:8], cur_hash[:8]
                )
            )
    else:
        git_hash_file.write_text(cur_hash)


def get_logger(model_dir: str, filename: str = "train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    log_dir = Path(model_dir)
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(log_dir / filename)
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger


class HParams:
    model_dir: str
    experiment_dir: str
    save_every_epoch: int
    name: str
    total_epoch: int
    pretrainG: str
    pretrainD: str
    version: str
    gpus: str
    train: "HParams"
    batch_size: int
    sample_rate: str
    if_f0: int
    if_latest: int
    save_every_weights: str
    if_cache_data_in_gpu: int
    data: "HParams"
    training_files: str

    def __init__(self, **kwargs: object) -> None:
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()
