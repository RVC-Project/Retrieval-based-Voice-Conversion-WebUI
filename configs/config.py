import argparse
import os
import sys
import json
from multiprocessing import cpu_count

import torch

try:
    import intel_extension_for_pytorch as ipex  # pylint: disable=import-error, unused-import

    if torch.xpu.is_available():
        from infer.modules.ipex import ipex_init

        ipex_init()
except Exception:  # pylint: disable=broad-exception-caught
    pass
import logging

logger = logging.getLogger(__name__)


version_config_list = [
    "v1/32k.json",
    "v1/40k.json",
    "v1/48k.json",
    "v2/48k.json",
    "v2/32k.json",
]


def singleton_variable(func):
    def wrapper(*args, **kwargs):
        if not wrapper.instance:
            wrapper.instance = func(*args, **kwargs)
        return wrapper.instance

    wrapper.instance = None
    return wrapper


@singleton_variable
class Config:
    def __init__(self):
        self.device = None
        self.is_half = False
        self.use_jit = True
        self.n_cpu = 0
        self.gpu_name = None
        self.json_config = self.load_config_json()
        self.gpu_mem = None
        self.all_device, self.all_device_index = self.get_all_device()
        (
            self.python_cmd,
            self.listen_port,
            self.iscolab,
            self.noparallel,
            self.noautoopen,
            self.dml,
            self.device_index
        ) = self.arg_parse()

        self.x_pad = None
        self.x_query = None
        self.x_center = None
        self.x_max = None

        if self.dml:
            for device_name,id in self.all_device_index.items():
                if "dml" in device_name.lower():
                    self.device_index = id
                    break
            
        self.set_device_by_index(self.device_index)



    @staticmethod
    def load_config_json() -> dict:
        d = {}
        for config_file in version_config_list:
            with open(f"configs/{config_file}", "r") as f:
                d[config_file] = json.load(f)
        return d

    @staticmethod
    def arg_parse() -> tuple:
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
        parser.add_argument(
            "--dml",
            action="store_true",
            help="torch_dml",
        )
        parser.add_argument(
            "--device_index",
            type=int,
            default=0,
            help="device index to be inferred or trained.",
        )
        cmd_opts = parser.parse_args()

        cmd_opts.port = cmd_opts.port if 0 <= cmd_opts.port <= 65535 else 7865

        return (
            cmd_opts.pycmd,
            cmd_opts.port,
            cmd_opts.colab,
            cmd_opts.noparallel,
            cmd_opts.noautoopen,
            cmd_opts.dml,
            cmd_opts.device_index
        )

    # has_mps is only available in nightly pytorch (for now) and MasOS 12.3+.
    # check `getattr` and try it for compatibility
    @classmethod
    def has_mps(self) -> bool:
        if not torch.backends.mps.is_available():
            return False
        try:
            torch.zeros(1).to(torch.device("mps"))
            return True
        except Exception:
            return False

    @classmethod
    def has_xpu(self) -> bool:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return True
        else:
            return False
    @classmethod
    def has_dml(self):
        try:
            import torch_directml
            return True
        except:
            return False

    def use_fp32_config(self):
        for config_file in version_config_list:
            self.json_config[config_file]["train"]["fp16_run"] = False
            with open(f"configs/{config_file}", "r") as f:
                strr = f.read().replace("true", "false")
            with open(f"configs/{config_file}", "w") as f:
                f.write(strr)
        with open("infer/modules/train/preprocess.py", "r") as f:
            strr = f.read().replace("3.7", "3.0")
        with open("infer/modules/train/preprocess.py", "w") as f:
            f.write(strr)
        print("overwrite preprocess and configs.json")

    @classmethod
    def get_all_device(self):
        gpu_counter=0
        device:list=[]
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device.extend([[f"{gpu_counter}: {torch.cuda.get_device_name(i)} (CUDA)", torch.device(f"cuda:{i}")]])
                gpu_counter+=1
        if self.has_xpu():
            for i in range(torch.xpu.device_count()):
                device.extend([[f"{gpu_counter}: {torch.cuda.get_device_name(i)} (XPU)", torch.device(f"xpu:{i}")]])
                gpu_counter+=1
        if self.has_mps():
            device.extend([[f"{gpu_counter}: MPS",torch.device("mps")]])
            gpu_counter+=1
        if self.has_dml():
            import torch_directml
            for i in range(torch_directml.device_count()):
                device.extend([[f"{gpu_counter}: {torch_directml.device_name(i)} (DML)",torch_directml.device(i)]])
                gpu_counter+=1
        device.append(["CPU",torch.device("cpu")])
        
        device_index:dict={device[i][0]:i for i in range(len(device))}
        return device, device_index

    def set_device_by_index(self,index:int):
        if index>=len(self.all_device)or index<0:
            raise ValueError("Out of index range.")

        device_name, self.device=self.all_device[index]
        logger.info(F"Use {device_name}")
        self.x_pad, self.x_query, self.x_center, self.x_max=self.device_config(self.device)


    def set_device_by_name(self,name:str):
        index=self.all_device_index[name]
        self.set_device_by_index(index)
        

    def device_config(self, device:torch.device) -> tuple:
        device_str = str(device).lower() 
        self.dml = False
        if "cuda" in device_str or "xpu" in device_str:
            if "xpu" in device_str:
                self.is_half = True
            i_device = int(device_str.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "P10" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                logger.info("GPU %s, force to fp32", self.gpu_name)
                self.is_half = False
                self.use_fp32_config()
            # else:
                # logger.info("Found GPU %s", self.gpu_name)
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            if self.gpu_mem<=4:
                with open("infer/modules/train/preprocess.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("infer/modules/train/preprocess.py", "w") as f:
                    f.write(strr)

        elif "mps" in device_str:
            self.is_half = False
            self.use_fp32_config()
        elif "privateuseone" in device_str:
            self.is_half = False
            self.dml = True
        else:
            self.is_half = False
            self.use_fp32_config()

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()
        
        if self.is_half:
            # 6G显存配置
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G显存配置
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem is not None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32


        # if ("cuda" in device_str or "xpu" in device_str ) and self.gpu_mem <= 4:
        #     with open("infer/modules/train/preprocess.py", "r") as f:
        #         strr = f.read().replace("per=3.7", "per=3.0")
        #     with open("infer/modules/train/preprocess.py", "w") as f:
        #         f.write(strr)
        # else:
        #     with open("infer/modules/train/preprocess.py", "r") as f:
        #         strr = f.read().replace("per=3.0", "per=3.7")
        #     with open("infer/modules/train/preprocess.py", "w") as f:
        #         f.write(strr)

        if self.dml:
            if (
                os.path.exists(
                    "runtime\Lib\site-packages\onnxruntime\capi\DirectML.dll"
                )
                == False
            ):
                try:
                    os.rename(
                        "runtime\Lib\site-packages\onnxruntime",
                        "runtime\Lib\site-packages\onnxruntime-cuda",
                    )
                except:
                    pass
                try:
                    os.rename(
                        "runtime\Lib\site-packages\onnxruntime-dml",
                        "runtime\Lib\site-packages\onnxruntime",
                    )
                except:
                    pass
        else:
            if (
                os.path.exists(
                    "runtime\Lib\site-packages\onnxruntime\capi\onnxruntime_providers_cuda.dll"
                )
                == False
            ):
                try:
                    os.rename(
                        "runtime\Lib\site-packages\onnxruntime",
                        "runtime\Lib\site-packages\onnxruntime-dml",
                    )
                except:
                    pass
                try:
                    os.rename(
                        "runtime\Lib\site-packages\onnxruntime-cuda",
                        "runtime\Lib\site-packages\onnxruntime",
                    )
                except:
                    pass
        print("is_half:%s, device:%s" % (self.is_half, self.device))
        return x_pad, x_query, x_center, x_max
