import os
import sys
import traceback
from pathlib import Path
from collections import OrderedDict

import torch

from i18n.i18n import I18nAuto

i18n = I18nAuto()

type WeightMap = dict[str, torch.Tensor]
type CheckpointValue = WeightMap | list[object] | str | int
type CheckpointDict = OrderedDict[str, CheckpointValue]


def savee(ckpt, sr, if_f0, name, epoch, version, hps):
    try:
        weights: WeightMap = {}
        for key in ckpt.keys():
            if "enc_q" in key:
                continue
            weights[key] = ckpt[key].half()
        opt: CheckpointDict = OrderedDict()
        opt["weight"] = weights
        opt["config"] = [
            hps.data.filter_length // 2 + 1,
            32,
            hps.model.inter_channels,
            hps.model.hidden_channels,
            hps.model.filter_channels,
            hps.model.n_heads,
            hps.model.n_layers,
            hps.model.kernel_size,
            hps.model.p_dropout,
            hps.model.resblock,
            hps.model.resblock_kernel_sizes,
            hps.model.resblock_dilation_sizes,
            hps.model.upsample_rates,
            hps.model.upsample_initial_channel,
            hps.model.upsample_kernel_sizes,
            hps.model.spk_embed_dim,
            hps.model.gin_channels,
            hps.data.sampling_rate,
        ]
        opt["info"] = "%sepoch" % epoch
        opt["sr"] = sr
        opt["f0"] = if_f0
        opt["version"] = version
        torch.save(opt, Path("assets/weights") / f"{name}.pth")
        return "Success."
    except:
        return traceback.format_exc()


def show_info(path: str):
    try:
        a = torch.load(Path(path), map_location="cpu", weights_only=False)
        return "模型信息:%s\n采样率:%s\n模型是否输入音高引导:%s\n版本:%s" % (
            a.get("info", "None"),
            a.get("sr", "None"),
            a.get("f0", "None"),
            a.get("version", "None"),
        )
    except:
        return traceback.format_exc()


def extract_small_model(path: str, name: str, sr, if_f0, info, version):
    try:
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        if "model" in ckpt:
            ckpt = ckpt["model"]
        weights: WeightMap = {}
        for key in ckpt.keys():
            if "enc_q" in key:
                continue
            weights[key] = ckpt[key].half()
        opt: CheckpointDict = OrderedDict()
        opt["weight"] = weights
        if sr == "40k":
            opt["config"] = [
                1025,
                32,
                192,
                192,
                768,
                2,
                6,
                3,
                0,
                "1",
                [3, 7, 11],
                [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                [10, 10, 2, 2],
                512,
                [16, 16, 4, 4],
                109,
                256,
                40000,
            ]
        elif sr == "48k":
            if version == "v1":
                opt["config"] = [
                    1025,
                    32,
                    192,
                    192,
                    768,
                    2,
                    6,
                    3,
                    0,
                    "1",
                    [3, 7, 11],
                    [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    [10, 6, 2, 2, 2],
                    512,
                    [16, 16, 4, 4, 4],
                    109,
                    256,
                    48000,
                ]
            else:
                opt["config"] = [
                    1025,
                    32,
                    192,
                    192,
                    768,
                    2,
                    6,
                    3,
                    0,
                    "1",
                    [3, 7, 11],
                    [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    [12, 10, 2, 2],
                    512,
                    [24, 20, 4, 4],
                    109,
                    256,
                    48000,
                ]
        elif sr == "32k":
            if version == "v1":
                opt["config"] = [
                    513,
                    32,
                    192,
                    192,
                    768,
                    2,
                    6,
                    3,
                    0,
                    "1",
                    [3, 7, 11],
                    [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    [10, 4, 2, 2, 2],
                    512,
                    [16, 16, 4, 4, 4],
                    109,
                    256,
                    32000,
                ]
            else:
                opt["config"] = [
                    513,
                    32,
                    192,
                    192,
                    768,
                    2,
                    6,
                    3,
                    0,
                    "1",
                    [3, 7, 11],
                    [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    [10, 8, 2, 2],
                    512,
                    [20, 16, 4, 4],
                    109,
                    256,
                    32000,
                ]
        if info == "":
            info = "Extracted model."
        opt["info"] = info
        opt["version"] = version
        opt["sr"] = sr
        opt["f0"] = int(if_f0)
        torch.save(opt, Path("assets/weights") / f"{name}.pth")
        return "Success."
    except:
        return traceback.format_exc()


def change_info(path: str, info, name):
    try:
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        ckpt["info"] = info
        if name == "":
            name = Path(path).name
        torch.save(ckpt, Path("assets/weights") / name)
        return "Success."
    except:
        return traceback.format_exc()


def merge(path1: str, path2: str, alpha1, sr, f0, info, name, version):
    try:

        def extract(ckpt):
            a = ckpt["model"]
            weights: WeightMap = {}
            for key in a.keys():
                if "enc_q" in key:
                    continue
                weights[key] = a[key]
            opt: CheckpointDict = OrderedDict()
            opt["weight"] = weights
            return opt

        ckpt1 = torch.load(Path(path1), map_location="cpu", weights_only=False)
        ckpt2 = torch.load(Path(path2), map_location="cpu", weights_only=False)
        cfg = ckpt1["config"]
        if "model" in ckpt1:
            ckpt1 = extract(ckpt1)
        else:
            ckpt1 = ckpt1["weight"]
        if "model" in ckpt2:
            ckpt2 = extract(ckpt2)
        else:
            ckpt2 = ckpt2["weight"]
        if sorted(list(ckpt1.keys())) != sorted(list(ckpt2.keys())):
            return "Fail to merge the models. The model architectures are not the same."
        weights: WeightMap = {}
        for key in ckpt1.keys():
            # try:
            if key == "emb_g.weight" and ckpt1[key].shape != ckpt2[key].shape:
                min_shape0 = min(ckpt1[key].shape[0], ckpt2[key].shape[0])
                weights[key] = (
                    alpha1 * (ckpt1[key][:min_shape0].float())
                    + (1 - alpha1) * (ckpt2[key][:min_shape0].float())
                ).half()
            else:
                weights[key] = (
                    alpha1 * (ckpt1[key].float()) + (1 - alpha1) * (ckpt2[key].float())
                ).half()
        opt: CheckpointDict = OrderedDict()
        opt["weight"] = weights
        # except:
        #     pdb.set_trace()
        opt["config"] = cfg
        """
        if(sr=="40k"):opt["config"] = [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10, 10, 2, 2], 512, [16, 16, 4, 4,4], 109, 256, 40000]
        elif(sr=="48k"):opt["config"] = [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10,6,2,2,2], 512, [16, 16, 4, 4], 109, 256, 48000]
        elif(sr=="32k"):opt["config"] = [513, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10, 4, 2, 2, 2], 512, [16, 16, 4, 4,4], 109, 256, 32000]
        """
        opt["sr"] = sr
        opt["f0"] = 1 if f0 == i18n("Yes") else 0
        opt["version"] = version
        opt["info"] = info
        torch.save(opt, Path("assets/weights") / f"{name}.pth")
        return "Success."
    except:
        return traceback.format_exc()
