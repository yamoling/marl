import asyncio
from threading import Thread
from websockets.server import serve, WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed
import psutil
import json
from marl.utils.others import list_gpus


async def send_system_info(websocket: WebSocketServerProtocol):
    try:
        while True:
            await asyncio.sleep(1)
            gpus = list_gpus()
            data = json.dumps(
                {
                    "cpus": psutil.cpu_percent(percpu=True),
                    "ram": psutil.virtual_memory().percent,
                    "gpus": [gpu.to_json() for gpu in gpus],
                }
            )
            await websocket.send(data)
    except ConnectionClosed:
        return


async def main(port: int):
    async with serve(send_system_info, "0.0.0.0", port):
        await asyncio.Future()  # run forever


def run(port: int):
    Thread(target=lambda: asyncio.run(main(port)), daemon=True).start()
