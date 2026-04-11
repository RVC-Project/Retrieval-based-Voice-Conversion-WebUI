"""Wrapper to run train.py with torch.load patch for PyTorch 2.6+ and Windows gloo fix."""
import os

import torch
import torch.multiprocessing

# --- Patch 1: torch.load weights_only for PyTorch 2.6+ ---
_orig_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)


torch.load = _patched_load

# --- Patch 2: Force 127.0.0.1 for gloo on Windows ---
os.environ["MASTER_ADDR"] = "127.0.0.1"

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    from infer.modules.train.train import main

    main()
