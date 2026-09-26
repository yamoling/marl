"""`GET /api/events`: Server-Sent Events."""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..services.events import stream
from . import HubDep

router = APIRouter(prefix="/api")


@router.get("/events")
async def events(hub: HubDep):
    """@ai-generated"""
    return StreamingResponse(
        stream(hub),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
