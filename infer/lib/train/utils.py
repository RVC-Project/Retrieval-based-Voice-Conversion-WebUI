import argparse
import glob
import json
import logging
import os
import subprocess
import sys
import shutil

import numpy as np
import torch
from scipy.io.wavfile import read

MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging


def load_checkpoint_d(checkpoint_path, combd, sbd, optimizer=None, load_opt=1):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")

    ##################
    def go(model, bkey):
        saved_state_dict = checkpoint_dict[bkey]
        if hasattr(model, "module"):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():  # 模型需要的shape
            try:
                new_state_dict[k] = saved_state_dict[k]
                if saved_state_dict[k].shape != state_dict[k].shape:
                    logger.warn(
                        "shape-%s-mismatch. need: %s, get: %s",
                        k,
                        state_dict[k].shape,
                        saved_state_dict[k].shape,
                    )  #
                    raise KeyError
            except:
                # logger.info(traceback.format_exc())
                logger.info("%s is not in the checkpoint", k)  # pretrain缺失的
                new_state_dict[k] = v  # 模型自带的随机值
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
    ):  ###加载不了，如果是空的的话，重新初始化，可能还会影响lr时间表的更新，因此在train文件最外围catch
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
def load_checkpoint(checkpoint_path, model, optimizer=None, load_opt=1):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")

    saved_state_dict = checkpoint_dict["model"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():  # 模型需要的shape
        try:
            new_state_dict[k] = saved_state_dict[k]
            if saved_state_dict[k].shape != state_dict[k].shape:
                logger.warn(
                    "shape-%s-mismatch|need-%s|get-%s",
                    k,
                    state_dict[k].shape,
                    saved_state_dict[k].shape,
                )  #
                raise KeyError
        except:
            # logger.info(traceback.format_exc())
            logger.info("%s is not in the checkpoint", k)  # pretrain缺失的
            new_state_dict[k] = v  # 模型自带的随机值
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)
    logger.info("Loaded model weights")

    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    if (
        optimizer is not None and load_opt == 1
    ):  ###加载不了，如果是空的的话，重新初始化，可能还会影响lr时间表的更新，因此在train文件最外围catch
        #   try:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    #   except:
    #     traceback.print_exc()
    logger.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
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


def save_checkpoint_d(combd, sbd, optimizer, learning_rate, iteration, checkpoint_path):
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


def summarize(
    writer,
    global_step,
    scalars={},
    histograms={},
    images={},
    audios={},
    audio_sampling_rate=22050,
):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    logger.debug(x)
    return x


def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot_alignment_to_numpy(alignment, info=None):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        alignment.transpose(), aspect="auto", origin="lower", interpolation="none"
    )
    fig.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def get_hparams(init=True):
    """
    todo:
      结尾七人组：
        保存频率、总epoch                     done
        bs                                    done
        pretrainG、pretrainD                  done
        卡号：os.en["CUDA_VISIBLE_DEVICES"]   done
        if_latest                             done
      模型：if_f0                             done
      采样率：自动选择config                  done
      是否缓存数据集进GPU:if_cache_data_in_gpu done

      -m:
        自动决定training_files路径,改掉train_nsf_load_pretrain.py里的hps.data.training_files    done
      -c不要了
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-se",
        "--save_every_epoch",
        type=int,
        required=True,
        help="checkpoint save frequency (epoch)",
    )
    parser.add_argument(
        "-te", "--total_epoch", type=int, required=True, help="total_epoch"
    )
    parser.add_argument(
        "-pg", "--pretrainG", type=str, default="", help="Pretrained Discriminator path"
    )
    parser.add_argument(
        "-pd", "--pretrainD", type=str, default="", help="Pretrained Generator path"
    )
    parser.add_argument("-g", "--gpus", type=str, default="0", help="split by -")
    parser.add_argument(
        "-bs", "--batch_size", type=int, required=True, help="batch size"
    )
    parser.add_argument(
        "-e", "--experiment_dir", type=str, required=True, help="experiment dir"
    )  # -m
    parser.add_argument(
        "-sr", "--sample_rate", type=str, required=True, help="sample rate, 32k/40k/48k"
    )
    parser.add_argument(
        "-sw",
        "--save_every_weights",
        type=str,
        default="0",
        help="save the extracted model in weights directory when saving checkpoints",
    )
    parser.add_argument(
        "-v", "--version", type=str, required=True, help="model version"
    )
    parser.add_argument(
        "-f0",
        "--if_f0",
        type=int,
        required=True,
        help="use f0 as one of the inputs of the model, 1 or 0",
    )
    parser.add_argument(
        "-l",
        "--if_latest",
        type=int,
        required=True,
        help="if only save the latest G/D pth file, 1 or 0",
    )
    parser.add_argument(
        "-c",
        "--if_cache_data_in_gpu",
        type=int,
        required=True,
        help="if caching the dataset in GPU memory, 1 or 0",
    )

    args = parser.parse_args()
    name = args.experiment_dir
    experiment_dir = os.path.join("./logs", args.experiment_dir)

    config_save_path = os.path.join(experiment_dir, "config.json")
    with open(config_save_path, "r") as f:
        config = json.load(f)

    hparams = HParams(**config)
    hparams.model_dir = hparams.experiment_dir = experiment_dir
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
    hparams.data.training_files = "%s/filelist.txt" % experiment_dir
    return hparams


def get_hparams_from_dir(model_dir):
    config_save_path = os.path.join(model_dir, "config.json")
    with open(config_save_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams


def get_hparams_from_file(config_path):
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams


def check_git_hash(model_dir):
    source_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(source_dir, ".git")):
        logger.warn(
            "{} is not a git repository, therefore hash value comparison will be ignored.".format(
                source_dir
            )
        )
        return

    cur_hash = subprocess.getoutput("git rev-parse HEAD")

    path = os.path.join(model_dir, "githash")
    if os.path.exists(path):
        saved_hash = open(path).read()
        if saved_hash != cur_hash:
            logger.warn(
                "git hash values are different. {}(saved) != {}(current)".format(
                    saved_hash[:8], cur_hash[:8]
                )
            )
    else:
        open(path, "w").write(cur_hash)


def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger


class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
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
