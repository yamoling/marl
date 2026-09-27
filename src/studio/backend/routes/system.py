"""System usage: `GET /api/system/specs` and `WS /api/system/ws`."""

import asyncio

import orjson
import psutil
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .. import settings
from ..errors import json_response
from ..security import is_allowed_origin, is_local_host
from ..services.system import system_info

router = APIRouter(prefix="/api")


@router.get("/system/specs")
def specs():
    """@ai-generated"""
    psutil.cpu_percent()  # The first call always returns 0
    return json_response(system_info())


@router.websocket("/system/ws")
async def system_ws(websocket: WebSocket):
    """Push the system usage every 0.5 s. Only local hosts and origins are accepted. @ai-generated"""
    host = websocket.headers.get("host", "")
    if not is_local_host(host) or not is_allowed_origin(websocket.headers.get("origin"), host):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    interval = getattr(websocket.app.state, "system_interval", settings.SYSTEM_WS_INTERVAL_S)
    try:
        while True:
            info = await asyncio.to_thread(system_info)
            await websocket.send_bytes(orjson.dumps(info))
            await asyncio.sleep(interval)
    except WebSocketDisconnect:
        return
