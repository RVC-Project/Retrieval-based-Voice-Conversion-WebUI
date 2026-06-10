# Electron UI for RVC Real-time Voice Conversion (macOS)

**Date:** 2026-06-10
**Status:** Draft — pending approval

## Problem

`run-realtime.sh` runs `tools/rvc_for_realtime.py`, but that file is a library:
its `__main__` block just prints "Real-time voice conversion ready" and exits.
The actual realtime UI is `gui_v1.py` (FreeSimpleGUI/tkinter), which is painful
on macOS and tangles the UI with the audio engine in one 1070-line file.

Goal: a native-feeling Electron app that serves as the UI for real-time voice
conversion on macOS, with full feature parity with `gui_v1.py`.

## Decisions made

- **UI scope:** full parity with `gui_v1.py` (all settings, both function modes,
  latency/inference-time readouts).
- **Launch style:** dev-style launch via an updated `run-realtime.sh`
  (no packaged .app for now).
- **Architecture:** Python owns all audio; Electron is a pure control panel.
  Audio never crosses the process boundary, so latency stays identical to
  gui_v1 (~90–170 ms).

## Architecture

```
┌─ Electron app ──────────────┐
│ renderer (UI) ⇄ main        │
│        │ WebSocket (JSON)   │
└────────┼────────────────────┘
         ▼ spawn + control
┌─ Python backend ────────────┐
│ FastAPI/uvicorn ws server   │
│ RealtimeVC engine           │
│  sounddevice ⇄ mic/speaker  │
│  RVC model (torch/MPS)      │
└─────────────────────────────┘
```

Two processes:

1. **Python backend** (`tools/realtime/`) — owns everything audio: the
   sounddevice full-duplex stream, the SOLA / noise-gate / buffer pipeline
   extracted from `gui_v1.py`, and the `infer.lib.rtrvc.RVC` model. Exposes a
   localhost WebSocket control server (FastAPI/uvicorn, already in the venv).
2. **Electron app** (`electron/`) — control panel in vanilla HTML/JS (no
   framework, no bundler). The main process spawns the backend with
   `.venv/bin/python`, waits for readiness, opens the window, and kills the
   backend on quit.

## Python backend

### `tools/realtime/engine.py`

`RealtimeVC` class — a faithful extraction of gui_v1's `GUI.start_vc`,
`start_stream`, `stop_stream`, `audio_callback`, `update_devices`,
`set_devices`, `get_device_samplerate`, `get_device_channels`, plus the
Harvest multiprocessing workers. Zero UI imports. Reports per-block inference
time and state changes through a callback supplied by the server.

### `tools/realtime/server.py`

FastAPI app bound to `127.0.0.1:6242` with a single WebSocket endpoint `/ws`.
JSON message protocol:

**Client → server**

| type             | payload                                  | effect |
|------------------|------------------------------------------|--------|
| `get_init`       | —                                        | returns hostapis, input/output devices, saved config |
| `update_devices` | `hostapi`                                | re-enumerates devices for that hostapi |
| `start`          | full config (paths, devices, all params) | validates, loads model, starts stream |
| `stop`           | —                                        | stops stream |
| `set_param`      | `key`, `value`                           | hot-updates a running stream |

Hot-updatable params (same set gui_v1 supports live): pitch, formant,
threshold, index rate, loudness factor (rms_mix_rate), f0 method, input/output
noise reduction, phase vocoder, vc/monitor function mode.

**Server → client**

| type      | payload |
|-----------|---------|
| `init`    | devices, hostapis, saved config |
| `started` | stream samplerate, computed algorithm delay (ms) |
| `stopped` | — |
| `stats`   | inference time per block (ms) |
| `error`   | human-readable message |
| `log`     | backend log line |

### Config persistence

Reads/writes the same `configs/inuse/config.json` with the same keys as
gui_v1, so settings carry over between the two UIs.

## Electron app

- `electron/main.js` — backend spawn/lifecycle, native file-open dialogs for
  `.pth` / `.index` files, macOS microphone permission via
  `systemPreferences.askForMediaAccess('microphone')`, error dialog with the
  last stderr lines if the backend process dies.
- `electron/preload.js` — contextBridge exposing only `pickFile()` and the
  backend port. The renderer talks to Python directly over WebSocket.
- `electron/renderer/` (`index.html`, `style.css`, `app.js`) — four sections
  mirroring gui_v1:
  1. **Model** — pth and index file pickers.
  2. **Devices** — hostapi dropdown, input/output device dropdowns, reload
     button, sample-rate source (model vs device).
  3. **Inference settings** — response threshold, pitch, formant (gender
     factor), index rate, loudness factor, f0 algorithm radio
     (pm / harvest / crepe / rmvpe / fcpe).
  4. **Performance** — block time, harvest process count, crossfade length,
     extra inference time, input/output noise reduction, phase vocoder.
  Plus a control bar: start / stop, output-vs-monitor toggle, algorithm
  latency (ms), inference time (ms). Controls enable/disable with stream
  state; hot-updatable params stay live while running.

## Launch & dependencies

- `run-realtime.sh` rewritten: activate `.venv`, install `sounddevice` if
  missing (absent from the venv today — it was only in the Windows realtime
  requirements files), `npm install` in `electron/` on first run, then launch
  Electron (which spawns the Python backend itself).
- Remove the misleading `__main__` stub in `tools/rvc_for_realtime.py`.

## Error handling

- Model-load failures, device errors, and stream exceptions surface as
  `error` messages rendered in a dismissible banner in the renderer.
- Missing pth/index paths are validated before start.
- Backend process exit → Electron dialog showing the last stderr lines.
- Port in use → backend tries successive ports and prints the chosen port on
  stdout for `main.js` to parse.

## Testing

- **Headless protocol smoke test** (`tools/realtime/test_protocol.py`): start
  the server, connect a WebSocket client, assert `get_init` returns devices,
  and assert `start` with a bogus model path yields a clean `error` message
  rather than a crash.
- **Manual e2e** on the dev machine: launch via `run-realtime.sh`, load a
  voice model, verify mic → converted output and the latency readouts.
  Requires a `.pth` voice model; `assets/weights/` is currently empty.
