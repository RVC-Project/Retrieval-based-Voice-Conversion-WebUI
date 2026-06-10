"""Import every module the refactor touches.

Run from repo root: .venv/bin/python tests/smoke_imports.py
"""

import importlib
import os
import sys

sys.path.insert(0, os.getcwd())

# infer.lib.rtrvc starts a multiprocessing.Manager at import time, so this
# script must guard __main__ (macOS uses the spawn start method).
MODULES = [
    "configs.config",
    "infer.lib.infer_pack.models",
    "infer.lib.infer_pack.models_onnx",
    "infer.lib.infer_pack.attentions",
    "infer.modules.vc.modules",
    "infer.modules.vc.pipeline",
    "infer.lib.rtrvc",
    "infer.lib.rmvpe",
    "infer.lib.train.process_ckpt",
    "tools.realtime.engine",
    "tools.realtime.harvest_worker",
    "i18n.i18n",
]

if __name__ == "__main__":
    for m in MODULES:
        importlib.import_module(m)
        print("ok", m)
    print("ALL OK")
