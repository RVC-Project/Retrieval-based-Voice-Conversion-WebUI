import os
import shutil

# Offline WebUI keeps the CUDA Graph implementation available, but remains
# eager by default. Set RVC_OFFLINE_CUDA_GRAPH=1 to opt in for benchmarking or
# controlled deployments.
_offline_cuda_graph = os.environ.get("RVC_OFFLINE_CUDA_GRAPH", "0") == "1"
os.environ["RVC_CUDA_GRAPH"] = "1" if _offline_cuda_graph else "0"

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("no_proxy", "localhost, 127.0.0.1, ::1")
os.environ.setdefault("weight_root", "assets/weights")
os.environ.setdefault("weight_pymss_root", "assets/pymss_weights")
os.environ.setdefault("index_root", "logs")
os.environ.setdefault("outside_index_root", "assets/indices")
os.environ.setdefault("rmvpe_root", "assets/rmvpe")

now_dir = os.getcwd()
tmp = os.path.join(now_dir, "TEMP")
os.makedirs(tmp, exist_ok=True)
os.environ["TEMP"] = tmp
for name in os.listdir(tmp):
    if name == "jieba.cache":
        continue
    path = os.path.join(tmp, name)
    delete = (
        os.remove if os.path.isfile(path) or os.path.islink(path) else shutil.rmtree
    )
    try:
        delete(path)
    except Exception as error:
        print(str(error))

from configs.config import Config, GPU_INDEX, GPU_INFOS, GPU_MEMORY, IS_GPU
from infer.vc.modules import VC
from tools.pymss_webui import PYMSS_MODEL_CHOICES, get_model_info, pymss_separate
from tools.file_io import read_text
from train.process_ckpt import (
    change_info,
    extract_small_model,
    merge,
    show_info,
)
from i18n.i18n import I18nAuto
import torch, platform
import numpy as np
import gradio as gr
import pathlib
import json
from time import sleep
from subprocess import Popen
from random import shuffle
import warnings
import traceback
import threading
import logging
import signal
import socket
import subprocess
import time


logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def find_available_port(start_port, host="0.0.0.0"):
    """Return the first bindable TCP port at or above ``start_port``."""
    if not 1 <= start_port <= 65535:
        raise ValueError(f"Port must be between 1 and 65535, got {start_port}.")

    for port in range(start_port, 65536):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind((host, port))
            return port
        except OSError:
            continue

    raise OSError(
        f"No available TCP port from {start_port} through 65535; WebUI was not started."
    )


def is_gradio_port_in_use_error(error, port):
    """Recognize Gradio's explicit-port conflict without hiding other launch errors."""
    return str(error).startswith(f"Port {port} is in use.")


def launch_webui_with_port_fallback(app, config):
    """Launch Gradio, increasing the requested port until startup succeeds."""
    next_port = config.listen_port
    queued_app = app.queue(concurrency_count=511, max_size=1022)
    while True:
        config.listen_port = find_available_port(next_port)
        if config.listen_port != next_port:
            logger.warning(
                "Port %s is occupied; trying port %s instead.",
                next_port,
                config.listen_port,
            )
        try:
            queued_app.launch(
                server_name="0.0.0.0",
                inbrowser=not config.noautoopen,
                server_port=config.listen_port,
                quiet=True,
            )
            return config.listen_port
        except OSError as error:
            if not is_gradio_port_in_use_error(error, config.listen_port):
                raise
            if config.listen_port == 65535:
                raise OSError(
                    "No available TCP port through 65535; WebUI was not started."
                ) from error
            logger.warning(
                "Port %s became occupied while Gradio was starting; trying the next port.",
                config.listen_port,
            )
            next_port = config.listen_port + 1

runtime_dirs = (
    os.path.join(now_dir, "logs"),
    os.environ["weight_root"],
    os.environ["weight_pymss_root"],
    os.environ["index_root"],
    os.environ["outside_index_root"],
    os.environ["rmvpe_root"],
    os.path.join(now_dir, "assets", "hubert_base"),
    os.path.join(now_dir, "assets", "pretrained"),
    os.path.join(now_dir, "assets", "pretrained_v2"),
)
for runtime_dir in runtime_dirs:
    os.makedirs(runtime_dir, exist_ok=True)
warnings.filterwarnings("ignore")
torch.manual_seed(114514)


config = Config()
vc = VC(config)


i18n = I18nAuto()
logger.info(i18n)
print(
    i18n("当前设备：%s | 推理精度：%s") % (config.device, config.dtype),
    flush=True,
)
# GPU filtering and precision rules are shared with inference/extraction/training.
gpu_infos = list(GPU_INFOS)
gpu_indices = sorted(GPU_INDEX)
if_gpu_ok = IS_GPU
if if_gpu_ok:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = max(1, int(min(GPU_MEMORY[i] for i in gpu_indices)) // 2)
else:
    gpu_info = i18n("很遗憾您这没有能用的显卡来支持您训练")
    default_batch_size = 1
gpus = "-".join(str(i) for i in gpu_indices)


class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"


weight_root = os.getenv("weight_root")
weight_pymss_root = os.getenv("weight_pymss_root")
outside_index_root = os.getenv("outside_index_root")

def weight_names():
    return sorted(
        name for name in os.listdir(weight_root) if name.endswith(".pth")
    )


def refresh_weight_choices(previous_names=None, force=False):
    current_names = tuple(weight_names())
    if force or current_names != previous_names:
        return current_names, change_choices()
    return current_names, {"__type__": "update"}


names = weight_names()
pymss_names = PYMSS_MODEL_CHOICES


def change_choices():
    return {"choices": weight_names(), "__type__": "update"}


def clean():
    return {"value": "", "__type__": "update"}


sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


TRAIN_TASK_LOCK = threading.Lock()
TRAIN_TASK = None


def button_update(value=None, variant=None, visible=None):
    update = {"__type__": "update"}
    if value is not None:
        update["value"] = value
    if variant is not None:
        update["variant"] = variant
    if visible is not None:
        update["visible"] = visible
    return update


def format_status(title, state, detail=""):
    lines = ["【%s】" % i18n(title), "%s：%s" % (i18n("状态"), i18n(state))]
    if detail:
        lines.extend(["", detail.strip()])
    return "\n".join(lines)


def format_workflow_status(step, detail="", completed_steps=None, state="运行中"):
    completed_steps = completed_steps or []
    detail = str(detail).strip()
    lines = []
    if completed_steps:
        lines.append("%s：" % i18n("已完成阶段"))
        lines.extend(
            "✓ %s：%s" % (i18n(completed_step), i18n("已成功"))
            for completed_step in completed_steps
        )
    if step:
        if lines:
            lines.append("")
        lines.append("%s：%s" % (i18n("当前阶段"), i18n(step)))
    if detail:
        lines.extend(["", detail])
    return format_status(
        "一键训练",
        state,
        "\n".join(lines),
    )


def read_log(path, max_lines=40):
    try:
        lines = [line.rstrip() for line in read_text(path, errors="ignore").splitlines()]
        lines = [line for line in lines if line.strip()]
        if len(lines) > max_lines:
            tail_count = max(0, max_lines - 1)
            omitted = len(lines) - tail_count
            tail = lines[-tail_count:] if tail_count else []
            lines = [i18n("……已省略前%s行，仅显示最新状态") % omitted]
            lines.extend(tail)
        return "\n".join(lines)
    except FileNotFoundError:
        return ""


def artifact_names(directory, suffix):
    if not os.path.isdir(directory):
        return set()
    return {
        name.split(".")[0]
        for name in os.listdir(directory)
        if name.lower().endswith(suffix)
    }


def validate_preprocess_outputs(exp_dir):
    exp_path = os.path.join(now_dir, "logs", exp_dir)
    gt_names = artifact_names(os.path.join(exp_path, "0_gt_wavs"), ".wav")
    wav16_names = artifact_names(os.path.join(exp_path, "1_16k_wavs"), ".wav")
    if not gt_names:
        raise RuntimeError(i18n("数据切分没有生成有效训练音频，请检查训练集和数据切分日志"))
    if not wav16_names:
        raise RuntimeError(i18n("数据切分没有生成16k音频，已停止后续特征提取和训练"))
    if not gt_names & wav16_names:
        raise RuntimeError(i18n("数据切分输出文件不匹配，已停止后续特征提取和训练"))


def validate_feature_outputs(exp_dir, version, if_f0):
    exp_path = os.path.join(now_dir, "logs", exp_dir)
    wav16_names = artifact_names(os.path.join(exp_path, "1_16k_wavs"), ".wav")
    feature_name = "3_feature256" if version == "v1" else "3_feature768"
    feature_names = artifact_names(os.path.join(exp_path, feature_name), ".npy")
    matched = wav16_names & feature_names
    if not feature_names or not matched:
        raise RuntimeError(i18n("HuBERT特征提取没有生成有效结果，已停止训练"))
    if if_f0:
        f0_names = artifact_names(os.path.join(exp_path, "2a_f0"), ".npy")
        f0nsf_names = artifact_names(os.path.join(exp_path, "2b-f0nsf"), ".npy")
        matched &= f0_names & f0nsf_names
        if not f0_names or not f0nsf_names or not matched:
            raise RuntimeError(i18n("F0提取没有生成有效结果，已停止训练"))
    return matched


def kill_process(process, process_name=""):
    if process is None or process.poll() is not None:
        return
    pid = process.pid
    if platform.system() == "Windows":
        subprocess.run(
            "taskkill /t /f /pid %s" % pid,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        for _ in range(10):
            if process.poll() is not None:
                break
            time.sleep(0.1)
        if process.poll() is None:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
    logger.info(i18n("%s进程已终止") % i18n(process_name))


def begin_train_task(name):
    global TRAIN_TASK
    with TRAIN_TASK_LOCK:
        if TRAIN_TASK is None:
            state = {
                "name": name,
                "processes": [],
                "stop_requested": False,
            }
            TRAIN_TASK = state
            return "start", state
        return "busy", TRAIN_TASK


def stop_train_task(name):
    with TRAIN_TASK_LOCK:
        if TRAIN_TASK is None:
            return (
                format_status(name, "未运行"),
                button_update(visible=True),
                button_update(visible=False),
            )
        if TRAIN_TASK["name"] != name:
            return (
                format_status(
                    name,
                    "无法停止",
                    i18n("%s运行中，请先停止该任务") % i18n(TRAIN_TASK["name"]),
                ),
                button_update(),
                button_update(),
            )
        state = TRAIN_TASK
        state["stop_requested"] = True
        processes = list(state["processes"])
    for process in processes:
        kill_process(process, name)
    return (
        format_status(name, "已停止"),
        button_update(visible=True),
        button_update(visible=False),
    )


def finish_train_task(state):
    global TRAIN_TASK
    with TRAIN_TASK_LOCK:
        if TRAIN_TASK is state:
            TRAIN_TASK = None


def train_task_stopped(state):
    with TRAIN_TASK_LOCK:
        return state["stop_requested"]


def start_train_process(state, cmd):
    kwargs = {"shell": True, "cwd": now_dir}
    if "train/train.py" in cmd.replace("\\", "/"):
        training_env = os.environ.copy()
        training_env["RVC_CUDA_GRAPH"] = "0"
        kwargs["env"] = training_env
    if platform.system() == "Windows":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    logger.info("%s: %s", i18n("执行命令"), cmd)
    process = Popen(cmd, **kwargs)
    with TRAIN_TASK_LOCK:
        state["processes"].append(process)
        stopped = state["stop_requested"]
    if stopped:
        kill_process(process, state["name"])
    return process


def wait_train_processes(
    state,
    processes,
    log_path=None,
    title="任务",
    format_output=True,
    watch_weights=False,
):
    last_snapshot = None
    last_emit_time = 0
    last_weight_names = tuple(weight_names()) if watch_weights else ()
    while any(process.poll() is None for process in processes):
        if train_task_stopped(state):
            for process in processes:
                kill_process(process, state["name"])
            break
        if log_path:
            snapshot = read_log(log_path)
            current_time = time.monotonic()
            current_weight_names = tuple(weight_names()) if watch_weights else ()
            weights_changed = watch_weights and current_weight_names != last_weight_names
            if (
                snapshot != last_snapshot
                or weights_changed
                or current_time - last_emit_time >= 5
            ):
                yield (
                    format_status(title, "运行中", snapshot)
                    if format_output
                    else snapshot
                )
                last_snapshot = snapshot
                last_emit_time = current_time
                last_weight_names = current_weight_names
        sleep(1)
    with TRAIN_TASK_LOCK:
        for process in processes:
            if process in state["processes"]:
                state["processes"].remove(process)
    if log_path:
        final_state = "已停止" if train_task_stopped(state) else "正在收尾"
        snapshot = read_log(log_path)
        yield (
            format_status(title, final_state, snapshot)
            if format_output
            else snapshot
        )
    if not train_task_stopped(state):
        failed = [process.returncode for process in processes if process.returncode != 0]
        if failed:
            raise RuntimeError(i18n("子进程执行失败，返回码：%s") % failed)


def run_preprocess_dataset(trainset_dir, exp_dir, sr, n_p, state, format_output=True):
    sr = sr_dict[sr]
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    log_path = "%s/logs/%s/preprocess.log" % (now_dir, exp_dir)
    with open(log_path, "w", encoding="utf8"):
        pass
    cmd = '"%s" train/preprocess.py "%s" %s %s "%s/logs/%s" %s %.1f' % (
        config.python_cmd,
        trainset_dir,
        sr,
        n_p,
        now_dir,
        exp_dir,
        config.noparallel,
        config.preprocess_per,
    )
    extract_start_time = time.time()
    requested_workers = max(int(n_p), 1)
    actual_workers = 1 if config.noparallel else requested_workers
    print(
        i18n(
            "数据提取开始：start_time=%.6f，请求并行数=%s，实际并行数上限=%s"
        )
        % (extract_start_time, requested_workers, actual_workers),
        flush=True,
    )
    try:
        process = start_train_process(state, cmd)
        yield from wait_train_processes(
            state, [process], log_path, "数据切分", format_output
        )
    finally:
        extract_end_time = time.time()
        print(
            i18n("数据提取结束：end_time=%.6f，总耗时=%.3f秒")
            % (extract_end_time, extract_end_time - extract_start_time),
            flush=True,
        )
    if not train_task_stopped(state):
        validate_preprocess_outputs(exp_dir)


def preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
    action, state = begin_train_task("数据切分")
    if action == "busy":
        yield (
            format_status(
                "数据切分",
                "等待中",
                i18n("%s运行中，请先停止该任务") % i18n(state["name"]),
            ),
            button_update(),
            button_update(),
        )
        return
    final_info = None
    try:
        yield (
            format_status("数据切分", "正在启动"),
            button_update(visible=False),
            button_update(visible=True),
        )
        for info in run_preprocess_dataset(trainset_dir, exp_dir, sr, n_p, state):
            yield info, button_update(visible=False), button_update(visible=True)
        if train_task_stopped(state):
            final_info = format_status("数据切分", "已停止")
    except Exception:
        final_info = format_status("数据切分", "失败", traceback.format_exc())
    finally:
        finish_train_task(state)
    if final_info is None:
        final_info = format_status("数据切分", "已完成")
    yield final_info, button_update(visible=True), button_update(visible=False)


def stop_preprocess_dataset():
    return stop_train_task("数据切分")


# but2.click(extract_f0,[gpus6,np7,f0method8,if_f0_3,trainset_dir4],[info2])
def run_extract_f0_feature(
    gpus,
    n_p,
    f0method,
    if_f0,
    exp_dir,
    version19,
    gpus_rmvpe,
    state,
    format_output=True,
):
    if f0method not in ("pm", "rmvpe"):
        raise ValueError(i18n("仅支持pm和rmvpe音高提取算法"))
    log_path = "%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir)
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    validate_preprocess_outputs(exp_dir)
    with open(log_path, "w", encoding="utf8"):
        pass

    if if_f0:
        processes = []
        rmvpe_devices = [gpu for gpu in gpus_rmvpe.split("-") if gpu != ""]
        if f0method == "pm" or (
            f0method == "rmvpe" and not rmvpe_devices and not config.dml
        ):
            cmd = (
                '"%s" train/dataset/extract_f0.py cpu "%s/logs/%s" %s %s'
                % (config.python_cmd, now_dir, exp_dir, n_p, f0method)
            )
            processes.append(start_train_process(state, cmd))
        elif rmvpe_devices:
            count = len(rmvpe_devices)
            for index, gpu in enumerate(rmvpe_devices):
                cmd = (
                    '"%s" train/dataset/extract_f0.py cuda %s %s %s "%s/logs/%s" %s'
                    % (
                        config.python_cmd,
                        count,
                        index,
                        gpu,
                        now_dir,
                        exp_dir,
                        config.is_half,
                    )
                )
                processes.append(start_train_process(state, cmd))
        else:
            cmd = (
                '"%s" train/dataset/extract_f0.py dml "%s/logs/%s"'
                % (config.python_cmd, now_dir, exp_dir)
            )
            processes.append(start_train_process(state, cmd))
        yield from wait_train_processes(
            state, processes, log_path, "F0提取", format_output
        )
        if train_task_stopped(state):
            return

        with open(log_path, "w", encoding="utf8"):
            pass

    feature_gpus = [gpu for gpu in gpus.split("-") if gpu != ""]
    processes = []
    if feature_gpus:
        count = len(feature_gpus)
        for index, gpu in enumerate(feature_gpus):
            cmd = (
                '"%s" train/dataset/extract_hubert_feature.py %s %s %s %s "%s/logs/%s" %s %s'
                % (
                    config.python_cmd,
                    config.device,
                    count,
                    index,
                    gpu,
                    now_dir,
                    exp_dir,
                    version19,
                    config.is_half,
                )
            )
            processes.append(start_train_process(state, cmd))
    else:
        cmd = (
            '"%s" train/dataset/extract_hubert_feature.py %s 1 0 "%s/logs/%s" %s %s'
            % (
                config.python_cmd,
                config.device,
                now_dir,
                exp_dir,
                version19,
                config.is_half,
            )
        )
        processes.append(start_train_process(state, cmd))
    yield from wait_train_processes(
        state, processes, log_path, "HuBERT特征", format_output
    )
    if not train_task_stopped(state):
        validate_feature_outputs(exp_dir, version19, if_f0)


def extract_f0_feature(gpus, n_p, f0method, if_f0, exp_dir, version19, gpus_rmvpe):
    action, state = begin_train_task("特征提取")
    if action == "busy":
        yield (
            format_status(
                "特征提取",
                "等待中",
                i18n("%s运行中，请先停止该任务") % i18n(state["name"]),
            ),
            button_update(),
            button_update(),
        )
        return
    final_info = None
    try:
        yield (
            format_status("特征提取", "正在启动"),
            button_update(visible=False),
            button_update(visible=True),
        )
        for info in run_extract_f0_feature(
            gpus, n_p, f0method, if_f0, exp_dir, version19, gpus_rmvpe, state
        ):
            yield info, button_update(visible=False), button_update(visible=True)
        if train_task_stopped(state):
            final_info = format_status("特征提取", "已停止")
    except Exception:
        final_info = format_status("特征提取", "失败", traceback.format_exc())
    finally:
        finish_train_task(state)
    if final_info is None:
        final_info = format_status("特征提取", "已完成")
    yield final_info, button_update(visible=True), button_update(visible=False)


def stop_extract_f0_feature():
    return stop_train_task("特征提取")

def get_pretrained_models(path_str, f0_str, sr2):
    if_pretrained_generator_exist = os.access(
        "assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if_pretrained_discriminator_exist = os.access(
        "assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if not if_pretrained_generator_exist:
        logger.warning(
            i18n("生成器预训练模型不存在，将不使用：assets/pretrained%s/%sG%s.pth"),
            path_str,
            f0_str,
            sr2,
        )
    if not if_pretrained_discriminator_exist:
        logger.warning(
            i18n("判别器预训练模型不存在，将不使用：assets/pretrained%s/%sD%s.pth"),
            path_str,
            f0_str,
            sr2,
        )
    return (
        (
            "assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2)
            if if_pretrained_generator_exist
            else ""
        ),
        (
            "assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2)
            if if_pretrained_discriminator_exist
            else ""
        ),
    )


def change_sr2(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    return get_pretrained_models(path_str, f0_str, sr2)


def change_version19(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    if sr2 == "32k" and version19 == "v1":
        sr2 = "40k"
    to_return_sr2 = (
        {"choices": ["40k", "48k"], "__type__": "update", "value": sr2}
        if version19 == "v1"
        else {"choices": ["40k", "48k", "32k"], "__type__": "update", "value": sr2}
    )
    f0_str = "f0" if if_f0_3 else ""
    return (
        *get_pretrained_models(path_str, f0_str, sr2),
        to_return_sr2,
    )


def change_f0(if_f0_3, sr2, version19):  # f0method8,pretrained_G14,pretrained_D15
    path_str = "" if version19 == "v1" else "_v2"
    return (
        {"visible": if_f0_3, "__type__": "update"},
        {"visible": if_f0_3, "__type__": "update"},
        *get_pretrained_models(path_str, "f0" if if_f0_3 == True else "", sr2),
    )


# but3.click(click_train,[exp_dir1,sr2,if_f0_3,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,gpus16])
def run_train_model(
    exp_dir1,
    sr2,
    if_f0_3,
    spk_id5,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
    state,
    format_output=True,
):
    # 生成filelist
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if if_f0_3:
        f0_dir = "%s/2a_f0" % (exp_dir)
        f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    if not names:
        raise RuntimeError(i18n("没有可用于训练的有效音频，请先完成数据切分和特征提取"))
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "w", encoding="utf8") as f:
        f.write("\n".join(opt))
    logger.debug(i18n("训练文件列表写入完成"))
    # 生成config#无需生成config
    # cmd = python_cmd + " train_nsf_sim_cache_sid_load_pretrain.py -e mi-test -sr 40k -f0 1 -bs 4 -g 0 -te 10 -se 5 -pg pretrained/f0G40k.pth -pd pretrained/f0D40k.pth -l 1 -c 0"
    logger.info(i18n("使用显卡：%s"), str(gpus16))
    if pretrained_G14 == "":
        logger.info(i18n("未使用生成器预训练模型"))
    if pretrained_D15 == "":
        logger.info(i18n("未使用判别器预训练模型"))
    if version19 == "v1" or sr2 == "40k":
        config_path = "v1/%s.json" % sr2
    else:
        config_path = "v2/%s.json" % sr2
    config_save_path = os.path.join(exp_dir, "config.json")
    if not pathlib.Path(config_save_path).exists():
        with open(config_save_path, "w", encoding="utf8") as f:
            json.dump(
                config.json_config[config_path],
                f,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            )
            f.write("\n")
    if gpus16:
        cmd = (
            '"%s" train/train.py -e "%s" -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
            % (
                config.python_cmd,
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                gpus16,
                total_epoch11,
                save_epoch10,
                "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                1 if if_save_latest13 == i18n("是") else 0,
                1 if if_cache_gpu17 == i18n("是") else 0,
                1 if if_save_every_weights18 == i18n("是") else 0,
                version19,
            )
        )
    else:
        cmd = (
            '"%s" train/train.py -e "%s" -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
            % (
                config.python_cmd,
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                total_epoch11,
                save_epoch10,
                "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                1 if if_save_latest13 == i18n("是") else 0,
                1 if if_cache_gpu17 == i18n("是") else 0,
                1 if if_save_every_weights18 == i18n("是") else 0,
                version19,
            )
        )
    logger.info("%s: %s", i18n("执行命令"), cmd)
    process = start_train_process(state, cmd)
    yield from wait_train_processes(
        state,
        [process],
        os.path.join(exp_dir, "train.log"),
        "模型训练",
        format_output,
        True,
    )


def click_train(
    exp_dir1,
    sr2,
    if_f0_3,
    spk_id5,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
):
    known_models = tuple(weight_names())
    action, state = begin_train_task("模型训练")
    if action == "busy":
        yield (
            format_status(
                "模型训练",
                "等待中",
                i18n("%s运行中，请先停止该任务") % i18n(state["name"]),
            ),
            button_update(),
            button_update(),
            button_update(),
        )
        return
    final_info = None
    try:
        yield (
            format_status("模型训练", "正在启动"),
            button_update(visible=False),
            button_update(visible=True),
            button_update(),
        )
        for info in run_train_model(
            exp_dir1,
            sr2,
            if_f0_3,
            spk_id5,
            save_epoch10,
            total_epoch11,
            batch_size12,
            if_save_latest13,
            pretrained_G14,
            pretrained_D15,
            gpus16,
            if_cache_gpu17,
            if_save_every_weights18,
            version19,
            state,
        ):
            known_models, model_update = refresh_weight_choices(known_models)
            yield (
                info,
                button_update(visible=False),
                button_update(visible=True),
                model_update,
            )
        if train_task_stopped(state):
            final_info = format_status("模型训练", "已停止")
    except Exception:
        final_info = format_status("模型训练", "失败", traceback.format_exc())
    finally:
        finish_train_task(state)
    if final_info is None:
        final_info = format_status("模型训练", "已完成")
    model_update = change_choices()
    yield (
        final_info,
        button_update(visible=True),
        button_update(visible=False),
        model_update,
    )


def stop_train_model():
    return stop_train_task("模型训练")


# but4.click(train_index, [exp_dir1], info3)
def run_train_index(exp_dir1, version19, state, format_output=True):
    exp_dir = os.path.join(now_dir, "logs", exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    log_path = os.path.join(exp_dir, "train_index.log")
    with open(log_path, "w", encoding="utf8"):
        pass
    cmd = (
        '"%s" train/train_index.py "%s" %s "%s" %s'
        % (
            config.python_cmd,
            exp_dir1,
            version19,
            outside_index_root,
            config.n_cpu,
        )
    )
    process = start_train_process(state, cmd)
    yield from wait_train_processes(
        state, [process], log_path, "索引训练", format_output
    )


def train_index(exp_dir1, version19):
    action, state = begin_train_task("索引训练")
    if action == "busy":
        yield (
            format_status(
                "索引训练",
                "等待中",
                i18n("%s运行中，请先停止该任务") % i18n(state["name"]),
            ),
            button_update(),
            button_update(),
        )
        return
    final_info = None
    try:
        yield (
            format_status("索引训练", "正在启动"),
            button_update(visible=False),
            button_update(visible=True),
        )
        for info in run_train_index(exp_dir1, version19, state):
            yield info, button_update(visible=False), button_update(visible=True)
        if train_task_stopped(state):
            final_info = format_status("索引训练", "已停止")
    except Exception:
        final_info = format_status("索引训练", "失败", traceback.format_exc())
    finally:
        finish_train_task(state)
    if final_info is None:
        final_info = format_status("索引训练", "已完成")
    yield final_info, button_update(visible=True), button_update(visible=False)


def stop_train_index():
    return stop_train_task("索引训练")

# but5.click(train1key, [exp_dir1, sr2, if_f0_3, trainset_dir4, spk_id5, gpus6, np7, f0method8, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17], info3)
def train1key(
    exp_dir1,
    sr2,
    if_f0_3,
    trainset_dir4,
    spk_id5,
    np7,
    f0method8,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
    gpus_rmvpe,
):
    known_models = tuple(weight_names())
    action, state = begin_train_task("一键训练")
    if action == "busy":
        yield (
            format_status(
                "一键训练",
                "等待中",
                i18n("%s运行中，请先停止该任务") % i18n(state["name"]),
            ),
            button_update(),
            button_update(),
            button_update(),
        )
        return

    completed_steps = []
    step = ""
    final_info = None
    running = True
    start_button = button_update(visible=False)
    stop_button = button_update(visible=True)
    try:
        yield (
            format_status("一键训练", "正在启动"),
            start_button,
            stop_button,
            button_update(),
        )

        step = "数据切分"
        yield (
            format_workflow_status(step, completed_steps=completed_steps),
            start_button,
            stop_button,
            button_update(),
        )
        for info in run_preprocess_dataset(
            trainset_dir4, exp_dir1, sr2, np7, state, False
        ):
            yield (
                format_workflow_status(step, info, completed_steps),
                start_button,
                stop_button,
                button_update(),
            )
        running = not train_task_stopped(state)
        if running:
            completed_steps.append(step)

        if running:
            step = "F0与HuBERT特征提取"
            yield (
                format_workflow_status(step, completed_steps=completed_steps),
                start_button,
                stop_button,
                button_update(),
            )
            for info in run_extract_f0_feature(
                gpus16,
                np7,
                f0method8,
                if_f0_3,
                exp_dir1,
                version19,
                gpus_rmvpe,
                state,
                False,
            ):
                yield (
                    format_workflow_status(step, info, completed_steps),
                    start_button,
                    stop_button,
                    button_update(),
                )
            running = not train_task_stopped(state)
            if running:
                completed_steps.append(step)

        if running:
            step = "模型训练"
            yield (
                format_workflow_status(step, completed_steps=completed_steps),
                start_button,
                stop_button,
                button_update(),
            )
            for info in run_train_model(
                exp_dir1,
                sr2,
                if_f0_3,
                spk_id5,
                save_epoch10,
                total_epoch11,
                batch_size12,
                if_save_latest13,
                pretrained_G14,
                pretrained_D15,
                gpus16,
                if_cache_gpu17,
                if_save_every_weights18,
                version19,
                state,
                False,
            ):
                known_models, model_update = refresh_weight_choices(known_models)
                yield (
                    format_workflow_status(step, info, completed_steps),
                    start_button,
                    stop_button,
                    model_update,
                )
            if not train_task_stopped(state):
                yield (
                    format_workflow_status(step, completed_steps=completed_steps),
                    start_button,
                    stop_button,
                    change_choices(),
                )
            running = not train_task_stopped(state)
            if running:
                completed_steps.append(step)

        if running:
            step = "索引训练"
            yield (
                format_workflow_status(step, completed_steps=completed_steps),
                start_button,
                stop_button,
                button_update(),
            )
            for info in run_train_index(exp_dir1, version19, state, False):
                yield (
                    format_workflow_status(step, info, completed_steps),
                    start_button,
                    stop_button,
                    button_update(),
                )
            running = not train_task_stopped(state)
            if running:
                completed_steps.append(step)

        if not running:
            final_info = format_workflow_status(
                step, completed_steps=completed_steps, state="已停止"
            )
        else:
            final_info = format_workflow_status(
                "", completed_steps=completed_steps, state="已完成"
            )
    except Exception:
        final_info = format_workflow_status(
            step,
            traceback.format_exc(),
            completed_steps,
            "失败",
        )
    finally:
        finish_train_task(state)
    model_update = change_choices()
    yield (
        final_info,
        button_update(visible=True),
        button_update(visible=False),
        model_update,
    )


def stop_train1key():
    return stop_train_task("一键训练")

#                    ckpt_path2.change(change_info_,[ckpt_path2],[sr__,if_f0__])
def change_info_(ckpt_path):
    if not os.path.exists(ckpt_path.replace(os.path.basename(ckpt_path), "train.log")):
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}
    try:
        info = eval(
            read_text(
                ckpt_path.replace(os.path.basename(ckpt_path), "train.log")
            )
            .strip("\n")
            .split("\n")[0]
            .split("\t")[-1]
        )
        sr, f0 = info["sample_rate"], info["if_f0"]
        version = "v2" if ("version" in info and info["version"] == "v2") else "v1"
        return sr, str(f0), version
    except Exception:
        traceback.print_exc()
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}


F0GPUVisible = IS_GPU


def change_f0_method(f0method8):
    if f0method8 == "rmvpe":
        visible = F0GPUVisible
    else:
        visible = False
    return {"visible": visible, "__type__": "update"}


with gr.Blocks(title="RVC WebUI") as app:
    gr.Markdown("## RVC WebUI")
    gr.Markdown(
        value=i18n(
            "本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>LICENSE</b>."
        )
    )
    with gr.Tabs():
        with gr.TabItem(i18n("模型推理")):
            with gr.Row():
                sid0 = gr.Dropdown(label=i18n("推理音色"), choices=sorted(names))
                with gr.Column():
                    refresh_button = gr.Button(
                        i18n("刷新音色列表"), variant="primary"
                    )
                    clean_button = gr.Button(i18n("卸载音色省显存"), variant="primary")
                spk_item = gr.Slider(
                    minimum=0,
                    maximum=2333,
                    step=1,
                    label=i18n("请选择说话人id"),
                    value=0,
                    visible=False,
                    interactive=True,
                )
                clean_button.click(
                    fn=clean, inputs=[], outputs=[sid0], api_name="infer_clean"
                )
            with gr.TabItem(i18n("单次推理")):
                with gr.Group():
                    with gr.Row():
                        with gr.Column():
                            with gr.Row(equal_height=True):
                                with gr.Column(scale=1, min_width=120):
                                    vc_transform0 = gr.Number(
                                        label=i18n("变调(整数, 半音数量, 升八度12降八度-12)"),
                                        value=0,
                                    )
                                with gr.Column(scale=2, min_width=200):
                                    f0method0 = gr.Radio(
                                        label=i18n("选择音高提取算法"),
                                        choices=["pm", "rmvpe", "fcpe"],
                                        value="rmvpe",
                                        interactive=True,
                                    )
                            input_audio0 = gr.Audio(
                                label=i18n("拖拽或点击上传待处理音频"),
                                source="upload",
                                type="filepath",
                                interactive=True,
                            )

                        with gr.Column():
                            resample_sr0 = gr.Slider(
                                minimum=0,
                                maximum=48000,
                                label=i18n("后处理重采样至最终采样率，0为不进行重采样"),
                                value=0,
                                step=1,
                                interactive=True,
                            )
                            rms_mix_rate0 = gr.Slider(
                                minimum=0,
                                maximum=1,
                                label=i18n(
                                    "输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络"
                                ),
                                value=0.25,
                                interactive=True,
                            )
                            protect0 = gr.Slider(
                                minimum=0,
                                maximum=0.5,
                                label=i18n(
                                    "保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果"
                                ),
                                value=0.33,
                                step=0.01,
                                interactive=True,
                            )
                            index_rate1 = gr.Slider(
                                minimum=0,
                                maximum=1,
                                label=i18n("检索特征占比"),
                                value=0.75,
                                interactive=True,
                            )
                            file_index1 = gr.Textbox(
                                label=i18n("特征检索库文件路径（选择模型后自动匹配，可手动修改）"),
                                placeholder="C:\\Users\\Desktop\\model_example.index",
                                interactive=True,
                            )
                            refresh_button.click(
                                fn=change_choices,
                                inputs=[],
                                outputs=sid0,
                                api_name="infer_refresh",
                            )
                with gr.Group():
                    with gr.Column():
                        but0 = gr.Button(i18n("转换"), variant="primary")
                        with gr.Row():
                            vc_output1 = gr.Textbox(label=i18n("输出信息"))
                            vc_output2 = gr.Audio(
                                label=i18n("输出音频(右下角三个点,点了可以下载)")
                            )

                        but0.click(
                            vc.vc_single,
                            [
                                spk_item,
                                input_audio0,
                                vc_transform0,
                                f0method0,
                                file_index1,
                                index_rate1,
                                resample_sr0,
                                rms_mix_rate0,
                                protect0,
                            ],
                            [vc_output1, vc_output2],
                            api_name="infer_convert",
                        )
            with gr.TabItem(i18n("批量推理")):
                gr.Markdown(
                    value=i18n(
                        "批量转换, 输入待转换音频文件夹, 或上传多个音频文件, 在指定文件夹(默认opt)下输出转换的音频. "
                    )
                )
                with gr.Row():
                    with gr.Column():
                        vc_transform1 = gr.Number(
                            label=i18n("变调(整数, 半音数量, 升八度12降八度-12)"),
                            value=0,
                        )
                        opt_input = gr.Textbox(
                            label=i18n("指定输出文件夹"), value="opt"
                        )
                        file_index3 = gr.Textbox(
                            label=i18n("特征检索库文件路径（选择模型后自动匹配，可手动修改）"),
                            value="",
                            interactive=True,
                        )
                        f0method1 = gr.Radio(
                            label=i18n("选择音高提取算法"),
                            choices=["pm", "rmvpe", "fcpe"],
                            value="rmvpe",
                            interactive=True,
                        )
                        format1 = gr.Radio(
                            label=i18n("导出文件格式"),
                            choices=["wav", "flac", "mp3", "m4a"],
                            value="wav",
                            interactive=True,
                        )

                    with gr.Column():
                        resample_sr1 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label=i18n("后处理重采样至最终采样率，0为不进行重采样"),
                            value=0,
                            step=1,
                            interactive=True,
                        )
                        rms_mix_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n(
                                "输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络"
                            ),
                            value=1,
                            interactive=True,
                        )
                        protect1 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label=i18n(
                                "保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果"
                            ),
                            value=0.33,
                            step=0.01,
                            interactive=True,
                        )
                        index_rate2 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("检索特征占比"),
                            value=1,
                            interactive=True,
                        )
                with gr.Row():
                    dir_input = gr.Textbox(
                        label=i18n(
                            "输入待处理音频文件夹路径(去文件管理器地址栏拷就行了)"
                        ),
                        placeholder="C:\\Users\\Desktop\\input_vocal_dir",
                    )
                    inputs = gr.File(
                        file_count="multiple",
                        label=i18n("也可批量输入音频文件, 二选一, 优先读文件夹"),
                    )

                with gr.Row():
                    but1 = gr.Button(i18n("转换"), variant="primary")
                    vc_output3 = gr.Textbox(label=i18n("输出信息"))

                    but1.click(
                        vc.vc_multi,
                        [
                            spk_item,
                            dir_input,
                            opt_input,
                            inputs,
                            vc_transform1,
                            f0method1,
                            file_index3,
                            index_rate2,
                            resample_sr1,
                            rms_mix_rate1,
                            protect1,
                            format1,
                        ],
                        [vc_output3],
                        api_name="infer_convert_batch",
                    )
                sid0.change(
                    fn=vc.get_vc,
                    inputs=[sid0, protect0, protect1],
                    outputs=[spk_item, protect0, protect1, file_index1, file_index3],
                    api_name="infer_change_voice",
                )
        with gr.TabItem(i18n("人声伴奏分离&去混响")):
            with gr.Group():
                gr.Markdown(
                    value=i18n(
                        "人声、伴奏与混响批量处理，使用pymss/MSST模型。"
                    )
                )
                with gr.Row():
                    with gr.Column():
                        dir_wav_input = gr.Textbox(
                            label=i18n("输入待处理音频文件夹路径"),
                            placeholder="C:\\Users\\Desktop\\todo-songs",
                        )
                        wav_inputs = gr.File(
                            file_count="multiple",
                            label=i18n("也可批量输入音频文件, 二选一, 优先读文件夹"),
                        )
                    with gr.Column():
                        model_choose = gr.Dropdown(
                            label=i18n("处理方式"),
                            choices=pymss_names,
                            value=pymss_names[0],
                            interactive=True,
                        )
                        model_info = gr.Textbox(
                            label=i18n("底层模型"),
                            value=get_model_info(pymss_names[0]),
                            interactive=False,
                        )
                        model_choose.change(
                            get_model_info,
                            [model_choose],
                            [model_info],
                            queue=False,
                        )
                        opt_vocal_root = gr.Textbox(
                            label=i18n("主结果文件夹"), value="opt"
                        )
                        opt_ins_root = gr.Textbox(
                            label=i18n("分离残余文件夹"), value="opt"
                        )
                        format0 = gr.Radio(
                            label=i18n("导出文件格式"),
                            choices=["wav", "flac", "mp3", "m4a"],
                            value="flac",
                            interactive=True,
                        )
                    but2 = gr.Button(i18n("转换"), variant="primary")
                    vc_output4 = gr.Textbox(label=i18n("输出信息"))
                    but2.click(
                        pymss_separate,
                        [
                            model_choose,
                            dir_wav_input,
                            opt_vocal_root,
                            wav_inputs,
                            opt_ins_root,
                            format0,
                        ],
                        [vc_output4],
                        api_name="uvr_convert",
                    )
        with gr.TabItem(i18n("训练")):
            gr.Markdown(
                value=i18n(
                    "step1: 填写实验配置. 实验数据放在logs下, 每个实验一个文件夹, 需手工输入实验名路径, 内含实验配置, 日志, 训练得到的模型文件. "
                )
            )
            with gr.Row():
                exp_dir1 = gr.Textbox(label=i18n("输入实验名"), value="mi-test")
                sr2 = gr.Radio(
                    label=i18n("目标采样率"),
                    choices=["40k", "48k"],
                    value="40k",
                    interactive=True,
                )
                if_f0_3 = gr.Radio(
                    label=i18n("模型是否带音高指导(唱歌一定要, 语音可以不要)"),
                    choices=[True, False],
                    value=True,
                    interactive=True,
                )
                version19 = gr.Radio(
                    label=i18n("版本"),
                    choices=["v1", "v2"],
                    value="v2",
                    interactive=True,
                    visible=True,
                )
                np7 = gr.Slider(
                    minimum=0,
                    maximum=config.n_cpu,
                    step=1,
                    label=i18n("提取音高和处理数据使用的CPU进程数"),
                    value=int(np.ceil(config.n_cpu / 1.5)),
                    interactive=True,
                )
            with gr.Group():  # 暂时单人的, 后面支持最多4人的#数据处理
                gr.Markdown(
                    value=i18n(
                        "step2a: 自动遍历训练文件夹下所有可解码成音频的文件并进行切片归一化, 在实验目录下生成2个wav文件夹; 暂时只支持单人训练. "
                    )
                )
                with gr.Row():
                    trainset_dir4 = gr.Textbox(
                        label=i18n("输入训练文件夹路径"),
                        value=i18n("E:\\语音音频+标注\\米津玄师\\src"),
                    )
                    spk_id5 = gr.Slider(
                        minimum=0,
                        maximum=4,
                        step=1,
                        label=i18n("请指定说话人id"),
                        value=0,
                        interactive=True,
                    )
                    but1 = gr.Button(i18n("处理数据"), variant="primary")
                    stop_but1 = gr.Button(
                        i18n("停止处理数据"), variant="stop", visible=False
                    )
                    info1 = gr.Textbox(label=i18n("输出信息"), value="")
                    but1.click(
                        preprocess_dataset,
                        [trainset_dir4, exp_dir1, sr2, np7],
                        [info1, but1, stop_but1],
                        api_name="train_preprocess",
                    )
                    stop_but1.click(
                        stop_preprocess_dataset,
                        [],
                        [info1, but1, stop_but1],
                        queue=False,
                    )
            with gr.Group():
                gr.Markdown(
                    value=i18n(
                        "step2b: 使用CPU提取音高(如果模型带音高), 使用GPU提取特征(选择卡号)"
                    )
                )
                with gr.Row():
                    with gr.Column():
                        gpus6 = gr.Textbox(
                            label=i18n(
                                "以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"
                            ),
                            value=gpus,
                            interactive=True,
                            visible=F0GPUVisible,
                        )
                        gpu_info9 = gr.Textbox(
                            label=i18n("显卡信息"), value=gpu_info, visible=F0GPUVisible
                        )
                    with gr.Column():
                        f0method8 = gr.Radio(
                            label=i18n("选择音高提取算法"),
                            choices=["pm", "rmvpe"],
                            value="rmvpe",
                            interactive=True,
                        )
                        gpus_rmvpe = gr.Textbox(
                            label=i18n(
                                "rmvpe卡号配置：以-分隔输入使用的不同进程卡号,例如0-0-1使用在卡0上跑2个进程并在卡1上跑1个进程"
                            ),
                            value="%s-%s" % (gpus, gpus),
                            interactive=True,
                            visible=F0GPUVisible,
                        )
                with gr.Row():
                    but2 = gr.Button(i18n("特征提取"), variant="primary")
                    stop_but2 = gr.Button(
                        i18n("停止特征提取"), variant="stop", visible=False
                    )
                info2 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8)
                f0method8.change(
                    fn=change_f0_method,
                    inputs=[f0method8],
                    outputs=[gpus_rmvpe],
                )
                but2.click(
                    extract_f0_feature,
                    [
                        gpus6,
                        np7,
                        f0method8,
                        if_f0_3,
                        exp_dir1,
                        version19,
                        gpus_rmvpe,
                    ],
                    [info2, but2, stop_but2],
                    api_name="train_extract_f0_feature",
                )
                stop_but2.click(
                    stop_extract_f0_feature,
                    [],
                    [info2, but2, stop_but2],
                    queue=False,
                )
            with gr.Group():
                gr.Markdown(value=i18n("step3: 填写训练设置, 开始训练模型和索引"))
                with gr.Row():
                    save_epoch10 = gr.Slider(
                        minimum=1,
                        maximum=50,
                        step=1,
                        label=i18n("保存频率save_every_epoch"),
                        value=5,
                        interactive=True,
                    )
                    total_epoch11 = gr.Slider(
                        minimum=2,
                        maximum=1200,
                        step=1,
                        label=i18n("总训练轮数total_epoch"),
                        value=20,
                        interactive=True,
                    )
                    batch_size12 = gr.Slider(
                        minimum=1,
                        maximum=40,
                        step=1,
                        label=i18n("每张显卡的batch_size"),
                        value=default_batch_size,
                        interactive=True,
                    )
                    if_save_latest13 = gr.Radio(
                        label=i18n("是否仅保存最新的ckpt文件以节省硬盘空间"),
                        choices=[i18n("是"), i18n("否")],
                        value=i18n("否"),
                        interactive=True,
                    )
                    if_cache_gpu17 = gr.Radio(
                        label=i18n(
                            "是否缓存所有训练集至显存. 10min以下小数据可缓存以加速训练, 大数据缓存会炸显存也加不了多少速"
                        ),
                        choices=[i18n("是"), i18n("否")],
                        value=i18n("否"),
                        interactive=True,
                    )
                    if_save_every_weights18 = gr.Radio(
                        label=i18n(
                            "是否在每次保存时间点将最终小模型保存至weights文件夹"
                        ),
                        choices=[i18n("是"), i18n("否")],
                        value=i18n("否"),
                        interactive=True,
                    )
                with gr.Row():
                    pretrained_G14 = gr.Textbox(
                        label=i18n("加载预训练底模G路径"),
                        value="assets/pretrained_v2/f0G40k.pth",
                        interactive=True,
                    )
                    pretrained_D15 = gr.Textbox(
                        label=i18n("加载预训练底模D路径"),
                        value="assets/pretrained_v2/f0D40k.pth",
                        interactive=True,
                    )
                    sr2.change(
                        change_sr2,
                        [sr2, if_f0_3, version19],
                        [pretrained_G14, pretrained_D15],
                    )
                    version19.change(
                        change_version19,
                        [sr2, if_f0_3, version19],
                        [pretrained_G14, pretrained_D15, sr2],
                    )
                    if_f0_3.change(
                        change_f0,
                        [if_f0_3, sr2, version19],
                        [f0method8, gpus_rmvpe, pretrained_G14, pretrained_D15],
                    )
                    gpus16 = gr.Textbox(
                        label=i18n(
                            "以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"
                        ),
                        value=gpus,
                        interactive=True,
                    )
                    but3 = gr.Button(i18n("训练模型"), variant="primary")
                    stop_but3 = gr.Button(
                        i18n("停止训练模型"), variant="stop", visible=False
                    )
                    but4 = gr.Button(i18n("训练特征索引"), variant="primary")
                    stop_but4 = gr.Button(
                        i18n("停止训练索引"), variant="stop", visible=False
                    )
                    but5 = gr.Button(i18n("一键训练"), variant="primary")
                    stop_but5 = gr.Button(
                        i18n("停止一键训练"), variant="stop", visible=False
                    )
                    info3 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=10)
                    but3.click(
                        click_train,
                        [
                            exp_dir1,
                            sr2,
                            if_f0_3,
                            spk_id5,
                            save_epoch10,
                            total_epoch11,
                            batch_size12,
                            if_save_latest13,
                            pretrained_G14,
                            pretrained_D15,
                            gpus16,
                            if_cache_gpu17,
                            if_save_every_weights18,
                            version19,
                        ],
                        [info3, but3, stop_but3, sid0],
                        api_name="train_start",
                    )
                    stop_but3.click(
                        stop_train_model,
                        [],
                        [info3, but3, stop_but3],
                        queue=False,
                    )
                    but4.click(
                        train_index,
                        [exp_dir1, version19],
                        [info3, but4, stop_but4],
                    )
                    stop_but4.click(
                        stop_train_index,
                        [],
                        [info3, but4, stop_but4],
                        queue=False,
                    )
                    but5.click(
                        train1key,
                        [
                            exp_dir1,
                            sr2,
                            if_f0_3,
                            trainset_dir4,
                            spk_id5,
                            np7,
                            f0method8,
                            save_epoch10,
                            total_epoch11,
                            batch_size12,
                            if_save_latest13,
                            pretrained_G14,
                            pretrained_D15,
                            gpus16,
                            if_cache_gpu17,
                            if_save_every_weights18,
                            version19,
                            gpus_rmvpe,
                        ],
                        [info3, but5, stop_but5, sid0],
                        api_name="train_start_all",
                    )
                    stop_but5.click(
                        stop_train1key,
                        [],
                        [info3, but5, stop_but5],
                        queue=False,
                    )

        with gr.TabItem(i18n("ckpt处理")):
            with gr.Group():
                gr.Markdown(value=i18n("模型融合, 可用于测试音色融合"))
                with gr.Row():
                    ckpt_a = gr.Textbox(
                        label=i18n("A模型路径"), value="", interactive=True
                    )
                    ckpt_b = gr.Textbox(
                        label=i18n("B模型路径"), value="", interactive=True
                    )
                    alpha_a = gr.Slider(
                        minimum=0,
                        maximum=1,
                        label=i18n("A模型权重"),
                        value=0.5,
                        interactive=True,
                    )
                with gr.Row():
                    sr_ = gr.Radio(
                        label=i18n("目标采样率"),
                        choices=["40k", "48k"],
                        value="40k",
                        interactive=True,
                    )
                    if_f0_ = gr.Radio(
                        label=i18n("模型是否带音高指导"),
                        choices=[i18n("是"), i18n("否")],
                        value=i18n("是"),
                        interactive=True,
                    )
                    info__ = gr.Textbox(
                        label=i18n("要置入的模型信息"),
                        value="",
                        max_lines=8,
                        interactive=True,
                    )
                    name_to_save0 = gr.Textbox(
                        label=i18n("保存的模型名不带后缀"),
                        value="",
                        max_lines=1,
                        interactive=True,
                    )
                    version_2 = gr.Radio(
                        label=i18n("模型版本型号"),
                        choices=["v1", "v2"],
                        value="v1",
                        interactive=True,
                    )
                with gr.Row():
                    but6 = gr.Button(i18n("融合"), variant="primary")
                    info4 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8)
                but6.click(
                    merge,
                    [
                        ckpt_a,
                        ckpt_b,
                        alpha_a,
                        sr_,
                        if_f0_,
                        info__,
                        name_to_save0,
                        version_2,
                    ],
                    info4,
                    api_name="ckpt_merge",
                )  # def merge(path1,path2,alpha1,sr,f0,info):
            with gr.Group():
                gr.Markdown(
                    value=i18n("修改模型信息(仅支持weights文件夹下提取的小模型文件)")
                )
                with gr.Row():
                    ckpt_path0 = gr.Textbox(
                        label=i18n("模型路径"), value="", interactive=True
                    )
                    info_ = gr.Textbox(
                        label=i18n("要改的模型信息"),
                        value="",
                        max_lines=8,
                        interactive=True,
                    )
                    name_to_save1 = gr.Textbox(
                        label=i18n("保存的文件名, 默认空为和源文件同名"),
                        value="",
                        max_lines=8,
                        interactive=True,
                    )
                with gr.Row():
                    but7 = gr.Button(i18n("修改"), variant="primary")
                    info5 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8)
                but7.click(
                    change_info,
                    [ckpt_path0, info_, name_to_save1],
                    info5,
                    api_name="ckpt_modify",
                )
            with gr.Group():
                gr.Markdown(
                    value=i18n("查看模型信息(仅支持weights文件夹下提取的小模型文件)")
                )
                with gr.Row():
                    ckpt_path1 = gr.Textbox(
                        label=i18n("模型路径"), value="", interactive=True
                    )
                    but8 = gr.Button(i18n("查看"), variant="primary")
                    info6 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8)
                but8.click(show_info, [ckpt_path1], info6, api_name="ckpt_show")
            with gr.Group():
                gr.Markdown(
                    value=i18n(
                        "模型提取(输入logs文件夹下大文件模型路径),适用于训一半不想训了模型没有自动提取保存小文件模型,或者想测试中间模型的情况"
                    )
                )
                with gr.Row():
                    ckpt_path2 = gr.Textbox(
                        label=i18n("模型路径"),
                        value="E:\\codes\\py39\\logs\\mi-test_f0_48k\\G_23333.pth",
                        interactive=True,
                    )
                    save_name = gr.Textbox(
                        label=i18n("保存名"), value="", interactive=True
                    )
                    sr__ = gr.Radio(
                        label=i18n("目标采样率"),
                        choices=["32k", "40k", "48k"],
                        value="40k",
                        interactive=True,
                    )
                    if_f0__ = gr.Radio(
                        label=i18n("模型是否带音高指导,1是0否"),
                        choices=["1", "0"],
                        value="1",
                        interactive=True,
                    )
                    version_1 = gr.Radio(
                        label=i18n("模型版本型号"),
                        choices=["v1", "v2"],
                        value="v2",
                        interactive=True,
                    )
                    info___ = gr.Textbox(
                        label=i18n("要置入的模型信息"),
                        value="",
                        max_lines=8,
                        interactive=True,
                    )
                    but9 = gr.Button(i18n("提取"), variant="primary")
                    info7 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8)
                    ckpt_path2.change(
                        change_info_, [ckpt_path2], [sr__, if_f0__, version_1]
                    )
                but9.click(
                    extract_small_model,
                    [ckpt_path2, save_name, sr__, if_f0__, info___, version_1],
                    info7,
                    api_name="ckpt_extract",
                )

        tab_faq = i18n("常见问题解答")
        with gr.TabItem(tab_faq):
            try:
                if tab_faq == "常见问题解答":
                    info = read_text("docs/cn/faq.md")
                else:
                    info = read_text("docs/en/faq_en.md")
                gr.Markdown(value=info)
            except Exception:
                gr.Markdown(traceback.format_exc())

    if config.iscolab:
        app.queue(concurrency_count=511, max_size=1022).launch(share=True)
    else:
        launch_webui_with_port_fallback(app, config)
