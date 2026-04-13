import asyncio
import orjson
import psutil
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from marl.utils import list_gpus


router = APIRouter()


def get_system_info():
    return {
        "cpus": psutil.cpu_percent(percpu=True),
        "ram": psutil.virtual_memory().percent,
        "gpus": list_gpus(),
    }


@router.websocket("/ws/system-info")
async def send_system_info(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await asyncio.sleep(0.5)
            data = orjson.dumps(get_system_info())
            await websocket.send_bytes(data)
    except WebSocketDisconnect:
        return
