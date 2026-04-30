import argparse
import os
import sys
import json
import shutil
from multiprocessing import cpu_count
from functools import wraps
from typing import TypeVar, TypedDict, cast

import torch

import logging

logger = logging.getLogger(__name__)


version_config_list: list[str] = [
    "v1/32k.json",
    "v1/40k.json",
    "v1/48k.json",
    "v2/48k.json",
    "v2/32k.json",
]

T = TypeVar("T")


class TrainConfig(TypedDict, total=False):
    fp16_run: bool


class VersionConfig(TypedDict, total=False):
    train: TrainConfig


_singleton_instances: dict[type[object], object] = {}


def singleton_class(cls: type[T]) -> type[T]:
    @wraps(cls)
    def wrapper(*args: object, **kwargs: object) -> T:
        if cls not in _singleton_instances:
            _singleton_instances[cls] = cls(*args, **kwargs)
        return cast(T, _singleton_instances[cls])

    return cast(type[T], wrapper)


# def singleton_variable(func: Callable[..., Any]) -> Callable[..., Any]:
#     def wrapper(*args, **kwargs):
#         if not wrapper.instance:
#             wrapper.instance = func(*args, **kwargs)
#         return wrapper.instance

#     wrapper.instance = None
#     return wrapper


@singleton_class
class Config:
    device: str
    is_half: bool
    use_jit: bool
    n_cpu: int
    gpu_name: str | None
    json_config: dict[str, VersionConfig]
    gpu_mem: int | None

    python_cmd: str
    listen_port: int
    iscolab: bool
    noparallel: bool
    noautoopen: bool

    instead: str
    preprocess_per: float
    x_pad: int
    x_query: int
    x_center: int
    x_max: int

    def __init__(self):
        self.device = "cuda:0"
        self.is_half: bool = True
        self.use_jit: bool = False
        self.n_cpu: int = 0
        self.gpu_name: str | None = None
        self.json_config: dict[str, VersionConfig] = self.load_config_json()
        self.gpu_mem: int | None = None
        (
            self.python_cmd,
            self.listen_port,
            self.iscolab,
            self.noparallel,
            self.noautoopen,
        ) = self.arg_parse()
        self.instead: str = ""
        self.preprocess_per: float = 3.7
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    @staticmethod
    def load_config_json() -> dict[str, VersionConfig]:
        d: dict[str, VersionConfig] = {}
        for config_file in version_config_list:
            p = f"configs/inuse/{config_file}"
            if not os.path.exists(p):
                shutil.copy(f"configs/{config_file}", p)
            with open(f"configs/inuse/{config_file}", "r") as f:
                d[config_file] = cast(VersionConfig, json.load(f))
        return d

    @staticmethod
    def arg_parse() -> tuple[str, int, bool, bool, bool]:
        exe = sys.executable or "python"
        parser = argparse.ArgumentParser()
        parser.add_argument("--port", type=int, default=7865, help="Listen port")
        parser.add_argument("--pycmd", type=str, default=exe, help="Python command")
        parser.add_argument("--colab", action="store_true", help="Launch in colab")
        parser.add_argument(
            "--noparallel", action="store_true", help="Disable parallel processing"
        )
        parser.add_argument(
            "--noautoopen",
            action="store_true",
            help="Do not open in browser automatically",
        )
        cmd_opts = parser.parse_args()

        cmd_opts.port = cmd_opts.port if 0 <= cmd_opts.port <= 65535 else 7865

        return (
            cmd_opts.pycmd,
            cmd_opts.port,
            cmd_opts.colab,
            cmd_opts.noparallel,
            cmd_opts.noautoopen,
        )

    # has_mps is only available in nightly pytorch (for now) and MasOS 12.3+.
    # check `getattr` and try it for compatibility
    @staticmethod
    def has_mps() -> bool:
        if not torch.backends.mps.is_available():
            return False
        try:
            torch.zeros(1).to(torch.device("mps"))
            return True
        except Exception:
            return False

    def use_fp32_config(self):
        for config_file in version_config_list:
            self.json_config[config_file].setdefault("train", {})["fp16_run"] = False
            with open(f"configs/inuse/{config_file}", "r") as f:
                strr = f.read().replace("true", "false")
            with open(f"configs/inuse/{config_file}", "w") as f:
                f.write(strr)
            logger.info("overwrite " + config_file)
        self.preprocess_per = 3.0
        logger.info("overwrite preprocess_per to %d" % (self.preprocess_per))

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "P10" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
                or "TITAN X" in self.gpu_name.upper()
                or "TITAN V" in self.gpu_name.upper()
                or "TITAN P" in self.gpu_name.upper()
            ):
                logger.info("Found GPU %s, force to fp32", self.gpu_name)
                self.is_half = False
                self.use_fp32_config()
            else:
                logger.info("Found GPU %s", self.gpu_name)
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            if self.gpu_mem <= 4:
                self.preprocess_per = 3.0
        elif self.has_mps():
            logger.info("No supported Nvidia GPU found")
            self.device = self.instead = "mps"
            self.is_half = False
            self.use_fp32_config()
        else:
            logger.info("No supported Nvidia GPU found")
            self.device = self.instead = "cpu"
            self.is_half = False
            self.use_fp32_config()

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            # VRAM >= 6GB: use x_pad=3, x_query=10, x_center=60, x_max=65
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # VRAM >= 4GB: use x_pad=1, x_query=6, x_center=38, x_max=41
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem is not None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32
        if self.instead:
            logger.info(f"Use {self.instead} instead")
        logger.info(
            "Half-precision floating-point: %s, device: %s"
            % (self.is_half, self.device)
        )
        return x_pad, x_query, x_center, x_max
