import argparse
import sys
import torch
from multiprocessing import cpu_count

global IsFp16Support
IsFp16Support = False

def use_fp32_config():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # Assuming you have only one GPU (index 0).
        if torch.cuda.get_device_capability(device)[0] >= 7:
            print("Fp16 supported!")
            IsFp16Support = True
            for config_file in ["32k.json", "40k.json", "48k.json"]:
                with open(f"configs/{config_file}", "r+") as f:
                    strr = f.read().replace("false", "true")
                    f.write(strr)
            with open("trainset_preprocess_pipeline_print.py", "r+") as f:
                strr = f.read().replace("3.0", "3.7")
                f.write(strr)
        else:
            print("fp16 unavailable! Using fp32.....")
            for config_file in ["32k.json", "40k.json", "48k.json"]:
                with open(f"configs/{config_file}", "r+") as f:
                    strr = f.read().replace("true", "false")
                    f.write(strr)
            with open("trainset_preprocess_pipeline_print.py", "r+") as f:
                strr = f.read().replace("3.7", "3.0")
                f.write(strr)
    else:
        print("CUDA is not available. Make sure you have an NVIDIA GPU and CUDA installed.")
    return (IsFp16Support, torch.cuda.get_device_capability(device)[0])


class Config:
    def __init__(self):
        self.device = "cuda:0"
        self.is_half = True
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        (
            self.python_cmd,
            self.listen_port,
            self.iscolab,
            self.noparallel,
            self.noautoopen,
            self.paperspace,
            self.is_cli,
        ) = self.arg_parse()
        
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

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
        parser.add_argument( # Fork Feature. Paperspace integration for web UI
            "--paperspace", action="store_true", help="Note that this argument just shares a gradio link for the web UI. Thus can be used on other non-local CLI systems."
        )
        parser.add_argument( # Fork Feature. Embed a CLI into the infer-web.py
            "--is_cli", action="store_true", help="Use the CLI instead of setting up a gradio UI. This flag will launch an RVC text interface where you can execute functions from infer-web.py!"
        )
        cmd_opts = parser.parse_args()

        cmd_opts.port = cmd_opts.port if 0 <= cmd_opts.port <= 65535 else 7865

        return (
            cmd_opts.pycmd,
            cmd_opts.port,
            cmd_opts.colab,
            cmd_opts.noparallel,
            cmd_opts.noautoopen,
            cmd_opts.paperspace,
            cmd_opts.is_cli,
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

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                print("Found GPU", self.gpu_name, ", force to fp32")
                self.is_half = False
            else:
                print("Found GPU", self.gpu_name)
                print(use_fp32_config())
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            if self.gpu_mem <= 4:
                with open("trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
        elif self.has_mps():
            print("No supported Nvidia GPU found, use MPS instead")
            self.device = "mps"
            self.is_half = False
            use_fp32_config()
        else:
            print("No supported Nvidia GPU found, use CPU instead")
            self.device = "cpu"
            self.is_half = False
            use_fp32_config()

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

        if self.gpu_mem != None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max
