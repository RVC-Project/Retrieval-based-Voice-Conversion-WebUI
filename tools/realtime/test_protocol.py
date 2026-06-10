"""Headless smoke test for the realtime control server protocol.

Run: .venv/bin/python tools/realtime/test_protocol.py

Starts the server as a subprocess, then checks:
  1. get_init returns device lists and the saved config
  2. start with a bogus model path returns a clean error (no crash)
  3. stop returns stopped
"""

import asyncio
import json
import os
import subprocess
import sys
import threading

NOW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SERVER = os.path.join(NOW_DIR, "tools", "realtime", "server.py")


def start_server():
    proc = subprocess.Popen(
        [sys.executable, SERVER, "--port", "6342"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=NOW_DIR,
    )
    port = None
    for line in proc.stdout:
        sys.stdout.write("[server] " + line)
        if line.startswith("REALTIME_SERVER_PORT="):
            port = int(line.strip().split("=", 1)[1])
            break
        if proc.poll() is not None:
            break
    if port is None:
        raise RuntimeError("Server exited before announcing its port")
    # keep draining stdout so the pipe never fills up
    threading.Thread(target=proc.stdout.read, daemon=True).start()
    return proc, port


async def connect_with_retry(uri):
    import websockets

    for _ in range(40):
        try:
            return await websockets.connect(uri)
        except OSError:
            await asyncio.sleep(0.5)
    raise RuntimeError("Could not connect to %s" % uri)


async def run_checks(port):
    ws = await connect_with_retry("ws://127.0.0.1:%d/ws" % port)
    try:
        await ws.send(json.dumps({"type": "get_init"}))
        init = json.loads(await ws.recv())
        assert init["type"] == "init", init
        assert isinstance(init["hostapis"], list) and init["hostapis"], init
        assert isinstance(init["input_devices"], list), init
        assert isinstance(init["output_devices"], list), init
        assert "pth_path" in init["config"], init
        print(
            "get_init OK: %d input / %d output devices"
            % (len(init["input_devices"]), len(init["output_devices"]))
        )

        bogus = dict(init["config"])
        bogus["pth_path"] = "/nonexistent/model.pth"
        bogus["index_path"] = "/nonexistent/model.index"
        await ws.send(json.dumps({"type": "start", "config": bogus}))
        resp = json.loads(await ws.recv())
        assert resp["type"] == "error", resp
        print("start with bogus path -> clean error: %s" % resp["message"])

        await ws.send(json.dumps({"type": "stop"}))
        resp = json.loads(await ws.recv())
        assert resp["type"] == "stopped", resp
        print("stop OK")
    finally:
        await ws.close()


def main():
    proc, port = start_server()
    try:
        asyncio.run(run_checks(port))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("PROTOCOL SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
