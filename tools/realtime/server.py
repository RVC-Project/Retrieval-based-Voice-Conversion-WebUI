"""WebSocket control server for the realtime VC engine.

Run: .venv/bin/python tools/realtime/server.py [--port 6242]

Announces the chosen port on stdout as REALTIME_SERVER_PORT=<port>.
Heavy imports happen inside main(): the multiprocessing spawn start
method re-imports this module in every Harvest worker, and those
workers must not pay for torch/fastapi imports.
"""

import argparse
import multiprocessing
import os
import sys

os.environ["OMP_NUM_THREADS"] = "4"
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

NOW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.path.join(NOW_DIR, "configs", "inuse", "config.json")

DEFAULT_CONFIG = {
    "pth_path": "",
    "index_path": "",
    "sg_hostapi": "",
    "sg_wasapi_exclusive": False,
    "sg_input_device": "",
    "sg_output_device": "",
    "sr_type": "sr_model",
    "threhold": -60,
    "pitch": 0,
    "formant": 0.0,
    "index_rate": 0,
    "rms_mix_rate": 0,
    "block_time": 0.25,
    "crossfade_length": 0.05,
    "extra_time": 2.5,
    "n_cpu": 4,
    "f0method": "rmvpe",
    "use_jit": False,
    "use_pv": False,
    "function": "vc",
}


def pick_port(host, start_port):
    import socket

    for port in range(start_port, start_port + 10):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(
        "No free port in range %d-%d" % (start_port, start_port + 9)
    )


def load_saved_config():
    import json

    try:
        with open(CONFIG_PATH, "r") as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {}
    merged = dict(DEFAULT_CONFIG)
    merged.update({k: v for k, v in data.items() if k in DEFAULT_CONFIG})
    return merged


def save_config(data):
    import json

    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    keep = {k: data.get(k, DEFAULT_CONFIG[k]) for k in DEFAULT_CONFIG}
    with open(CONFIG_PATH, "w") as f:
        json.dump(keep, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6242)
    args = parser.parse_args()
    # configs.config.Config runs its own argparse; hide our args from it
    sys.argv = sys.argv[:1]

    os.chdir(NOW_DIR)
    if NOW_DIR not in sys.path:
        sys.path.append(NOW_DIR)

    n_cpu = min(multiprocessing.cpu_count(), 8)
    inp_q = multiprocessing.Queue()
    opt_q = multiprocessing.Queue()
    from tools.realtime.harvest_worker import Harvest

    for _ in range(n_cpu):
        p = Harvest(inp_q, opt_q)
        p.daemon = True
        p.start()

    import asyncio
    import json
    import traceback

    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect

    from tools.realtime.engine import RealtimeVC

    app = FastAPI()
    clients = set()
    state = {"loop": None}

    def push_status(msg):
        loop = state["loop"]
        if loop is not None:
            loop.call_soon_threadsafe(_broadcast, msg)

    def _broadcast(msg):
        text = json.dumps(msg)
        for ws in list(clients):
            asyncio.ensure_future(_send_safe(ws, text))

    async def _send_safe(ws, text):
        try:
            await ws.send_text(text)
        except Exception:
            clients.discard(ws)

    engine = RealtimeVC(inp_q, opt_q, status_callback=push_status)

    def devices_payload():
        return {
            "hostapis": engine.hostapis,
            "input_devices": engine.input_devices,
            "output_devices": engine.output_devices,
        }

    async def handle(msg):
        mtype = msg.get("type")
        if mtype == "get_init":
            saved = load_saved_config()
            if saved["sg_hostapi"] in engine.hostapis:
                engine.update_devices(hostapi_name=saved["sg_hostapi"])
            payload = {"type": "init", "config": saved, "n_cpu_max": n_cpu}
            payload.update(devices_payload())
            return payload
        if mtype == "update_devices":
            engine.update_devices(hostapi_name=msg.get("hostapi"))
            payload = {"type": "devices"}
            payload.update(devices_payload())
            return payload
        if mtype == "start":
            data = msg.get("config", {})
            try:
                result = await asyncio.to_thread(engine.start, data)
            except Exception as exc:
                traceback.print_exc()
                return {"type": "error", "message": str(exc)}
            save_config(data)
            return {"type": "started", **result}
        if mtype == "stop":
            await asyncio.to_thread(engine.stop)
            return {"type": "stopped"}
        if mtype == "set_param":
            try:
                updates = engine.set_param(msg.get("key"), msg.get("value"))
            except Exception as exc:
                return {"type": "error", "message": str(exc)}
            if updates:
                return {"type": "param_updated", **updates}
            return None
        return {"type": "error", "message": "Unknown message type: %r" % mtype}

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        state["loop"] = asyncio.get_running_loop()
        clients.add(ws)
        try:
            while True:
                msg = json.loads(await ws.receive_text())
                resp = await handle(msg)
                if resp is not None:
                    await ws.send_text(json.dumps(resp))
        except WebSocketDisconnect:
            pass
        finally:
            clients.discard(ws)

    port = pick_port(args.host, args.port)
    print("REALTIME_SERVER_PORT=%d" % port, flush=True)
    uvicorn.run(app, host=args.host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
