import logging
import os
import threading
import time
from collections import OrderedDict

import torch


logger = logging.getLogger(__name__)

ENV_NAME = "RVC_CUDA_GRAPH"
MAX_CACHE_ENV = "RVC_CUDA_GRAPH_MAX_CACHE"
_probe_lock = threading.Lock()
_probe_result = None


def _device_type(device):
    if isinstance(device, torch.device):
        return device.type
    return str(device).split(":", 1)[0].lower()


def _cuda_device(device):
    parsed = device if isinstance(device, torch.device) else torch.device(device)
    if parsed.index is None:
        parsed = torch.device("cuda", torch.cuda.current_device())
    return parsed


def _clone_output(value):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone_output(item) for item in value)
    if isinstance(value, list):
        return [_clone_output(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_output(item) for key, item in value.items()}
    return value


def detect_cuda_graph_support(device):
    if _device_type(device) != "cuda" or not torch.cuda.is_available():
        return False
    if not hasattr(torch.cuda, "CUDAGraph") or not hasattr(torch.cuda, "graph"):
        return False
    cuda_device = _cuda_device(device)
    try:
        with torch.cuda.device(cuda_device):
            current = torch.cuda.current_stream(cuda_device)
            warmup = torch.cuda.Stream(device=cuda_device)
            warmup.wait_stream(current)
            with torch.cuda.stream(warmup):
                probe = torch.arange(32, device=cuda_device, dtype=torch.float32)
                for _ in range(3):
                    expected = probe.square().add_(1)
            current.wait_stream(warmup)
            torch.cuda.synchronize(cuda_device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = probe.square() + 1
            probe.copy_(torch.arange(32, device=cuda_device, dtype=torch.float32))
            graph.replay()
            torch.cuda.synchronize(cuda_device)
            valid = torch.equal(
                captured.cpu(), torch.arange(32, dtype=torch.float32).square() + 1
            )
            del captured, expected, graph, probe
            return bool(valid)
    except Exception:
        logger.exception("CUDA Graph support probe failed on %s", cuda_device)
        return False


def configure_cuda_graph(device):
    global _probe_result
    explicit = os.environ.get(ENV_NAME)
    if explicit in {"0", "1"}:
        if explicit == "0":
            return False
        if _device_type(device) != "cuda":
            os.environ[ENV_NAME] = "0"
            return False
    with _probe_lock:
        if _probe_result is None:
            _probe_result = detect_cuda_graph_support(device)
        os.environ[ENV_NAME] = "1" if _probe_result else "0"
    return bool(_probe_result)


def cuda_graph_enabled(device):
    return (
        os.environ.get(ENV_NAME) == "1"
        and _device_type(device) == "cuda"
        and torch.cuda.is_available()
    )


def _tensor_signature(tensor):
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
        bool(tensor.requires_grad),
    )


class _CapturedCall:
    def __init__(self, function, inputs):
        started = time.perf_counter()
        self.lock = threading.RLock()
        self.inputs = tuple(torch.empty_like(value) for value in inputs)
        for static, value in zip(self.inputs, inputs):
            static.copy_(value)
        device = self.inputs[0].device
        current = torch.cuda.current_stream(device)
        warmup = torch.cuda.Stream(device=device)
        warmup.wait_stream(current)
        with torch.cuda.stream(warmup), torch.no_grad():
            for _ in range(3):
                output = function(*self.inputs)
        current.wait_stream(warmup)
        torch.cuda.synchronize(device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.output = function(*self.inputs)
        self.capture_ms = (time.perf_counter() - started) * 1000.0
        self.done_event = None
        del output

    def replay(self, inputs):
        with self.lock:
            stream = torch.cuda.current_stream(self.inputs[0].device)
            if self.done_event is not None:
                stream.wait_event(self.done_event)
            for static, value in zip(self.inputs, inputs):
                static.copy_(value, non_blocking=True)
            self.graph.replay()
            output = _clone_output(self.output)
            self.done_event = torch.cuda.Event(blocking=False)
            self.done_event.record(stream)
            return output


class _GraphCache:
    def __init__(self):
        self.entries = OrderedDict()
        self.failures = set()
        self.lock = threading.RLock()
        self.capture_count = 0
        self.replay_count = 0
        self.fallback_count = 0
        self.eviction_count = 0
        self.capture_ms = 0.0

    def run(self, key, function, inputs):
        signature = key + tuple(_tensor_signature(value) for value in inputs)
        with self.lock:
            if signature in self.failures:
                self.fallback_count += 1
                return function(*inputs)
            entry = self.entries.get(signature)
            if entry is None:
                try:
                    entry = _CapturedCall(function, inputs)
                    self.entries[signature] = entry
                    self.capture_count += 1
                    self.capture_ms += entry.capture_ms
                    max_entries = max(1, int(os.environ.get(MAX_CACHE_ENV, "8")))
                    while len(self.entries) > max_entries:
                        self.entries.popitem(last=False)
                        self.eviction_count += 1
                except Exception:
                    self.failures.add(signature)
                    self.fallback_count += 1
                    logger.exception("CUDA Graph capture failed for %s; using eager", key)
                    return function(*inputs)
            else:
                self.entries.move_to_end(signature)
        output = entry.replay(inputs)
        with self.lock:
            self.replay_count += 1
        return output


def run_cuda_graph(owner, namespace, function, *inputs):
    if not inputs or not cuda_graph_enabled(inputs[0].device):
        return function(*inputs)
    cache = getattr(owner, "_rvc_cuda_graph_cache", None)
    if cache is None:
        cache = _GraphCache()
        setattr(owner, "_rvc_cuda_graph_cache", cache)
    return cache.run((str(namespace),), function, tuple(inputs))


def clear_cuda_graph_cache(owner):
    cache = getattr(owner, "_rvc_cuda_graph_cache", None)
    if cache is not None:
        cache.entries.clear()
        cache.failures.clear()
        delattr(owner, "_rvc_cuda_graph_cache")


def get_cuda_graph_stats(owner):
    cache = getattr(owner, "_rvc_cuda_graph_cache", None)
    if cache is None:
        return {
            "entries": 0,
            "failures": 0,
            "captures": 0,
            "replays": 0,
            "fallbacks": 0,
            "evictions": 0,
            "capture_ms": 0.0,
        }
    with cache.lock:
        return {
            "entries": len(cache.entries),
            "failures": len(cache.failures),
            "captures": cache.capture_count,
            "replays": cache.replay_count,
            "fallbacks": cache.fallback_count,
            "evictions": cache.eviction_count,
            "capture_ms": cache.capture_ms,
        }
