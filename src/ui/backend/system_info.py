import asyncio
from threading import Thread
from websockets.exceptions import ConnectionClosed
from websockets import serve
import orjson
import psutil
from marl.utils import list_gpus


stop = False


def get_system_info():
    return {
        "cpus": psutil.cpu_percent(percpu=True),
        "ram": psutil.virtual_memory().percent,
        "gpus": list_gpus(),
    }


async def send_system_info(websocket):
    try:
        while not stop:
            await asyncio.sleep(1)
            data = orjson.dumps(get_system_info())
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
