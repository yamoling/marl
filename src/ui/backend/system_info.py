import asyncio
from threading import Thread
from serde.json import to_json
from websockets.server import serve, WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed
import psutil
from marl.utils import list_gpus


stop = False


def get_system_info():
    return {
        "cpus": psutil.cpu_percent(percpu=True),
        "ram": psutil.virtual_memory().percent,
        "gpus": list_gpus(),
    }


async def send_system_info(websocket: WebSocketServerProtocol):
    try:
        while not stop:
            await asyncio.sleep(1)
            data = to_json(get_system_info())
            await websocket.send(data)
    except ConnectionClosed:
        return


async def main(port: int):
    print(f"Starting system info server on port {port}")
    async with serve(send_system_info, "", port):
        await asyncio.Future()  # run forever


def run(port: int):
    t = Thread(target=lambda: asyncio.run(main(port)), daemon=True)
    t.start()
