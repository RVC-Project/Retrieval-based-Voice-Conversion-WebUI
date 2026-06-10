"""Builds the full Gradio app without launching.

Run from repo root: .venv/bin/python tests/webui_build.py
"""

import os
import sys

sys.path.insert(0, os.getcwd())

if __name__ == "__main__":
    from web.app import build_app

    app = build_app()
    assert app is not None
    print("blocks:", len(app.blocks))
    print("WEBUI BUILD OK")
