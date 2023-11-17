import asyncio
from threading import Thread
from serde.json import to_json
from websockets.server import serve, WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed
import psutil
from marl.utils.others import list_gpus


def get_system_info():
    return {
        "cpus": psutil.cpu_percent(percpu=True),
        "ram": psutil.virtual_memory().percent,
        "gpus": list_gpus(),
    }


async def send_system_info(websocket: WebSocketServerProtocol):
    print("sending system info")
    try:
        while True:
            await asyncio.sleep(1)
            data = to_json(get_system_info())
            await websocket.send(data)
    except ConnectionClosed:
        return


async def main(port: int):
    async with serve(send_system_info, "0.0.0.0", port):
        await asyncio.Future()  # run forever


def run(port: int):
    Thread(target=lambda: asyncio.run(main(port)), daemon=True).start()
