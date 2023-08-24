import asyncio
import json
from threading import Thread

from vc_infer_utils import vc_single_json
from src.utils import get_voice_weights

from websockets.server import serve


async def _handle_generate(websocket, message):
    message['result'] = vc_single_json(message)

    # wipe the audio data in the response by default
    if not message.get('send_audio', False) and message['result'].get('audio'):
        message['result'].pop('audio')

    await websocket.send(json.dumps(message))

async def _handle_get_voice_weights(websocket, message):
    message['result'] = get_voice_weights()
    await websocket.send(json.dumps(message))

async def _handle_message(websocket, message):
    if message.get('action') and message['action'] == 'generate':
        await _handle_generate(websocket, message)
    if message.get('action') and message['action'] == 'get_voice_weights':
        await _handle_get_voice_weights(websocket, message)
    else:
        print(message)


async def _handle_connection(websocket, path):
    print("websocket: client connected")

    async for message in websocket:
        try:
            await _handle_message(websocket, json.loads(message))
        except ValueError:
            print("websocket: malformed json received")


async def _run(host: str, port: int):
    print("websocket: server started")

    async with serve(_handle_connection, host, port, ping_interval=None):
        await asyncio.Future()  # run forever


def _run_server(listen_address: str, port: int):
    asyncio.run(_run(host=listen_address, port=port))

def start_websocket_server(listen_address: str, port: int):
    Thread(target=_run_server, args=[listen_address, port], daemon=True).start()
