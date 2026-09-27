"""Run routes: replay, stop and restart."""

from http import HTTPStatus

from fastapi import APIRouter, Response

from ..errors import bad_request
from ..services import launcher, replay, runs
from . import HubDep, JsonBody, LibraryDep, run_record

router = APIRouter(prefix="/api")


@router.get("/runs/{run_id:path}/replay")
def replay_episode(run_id: str, step: int, test: int, library: LibraryDep, only_saved_actions: bool = False):
    """
    `ReplayEpisode` JSON (same serialization as the old UI). Sync route: runs in a worker thread.
    409 if the experiment is not replayable.

    @ai-generated
    """
    record, run = run_record(library, run_id)
    body = replay.replay(library, record, run, step, test, only_saved_actions)
    return Response(body, media_type="application/json")


@router.post("/runs/{run_id:path}/stop")
def stop_run(run_id: str, library: LibraryDep, hub: HubDep):
    """@ai-generated"""
    record, run = run_record(library, run_id)
    runs.stop_run(library, record, run)
    hub.notify_changed(record.id)
    return Response(status_code=HTTPStatus.NO_CONTENT)


@router.post("/runs/{run_id:path}/restart")
def restart_run(run_id: str, library: LibraryDep, hub: HubDep, body: JsonBody = None):
    """Restart a CANCELLED/CREATED run of a launchable experiment: 202, else 409. @ai-generated"""
    record, run = run_record(library, run_id)
    if body is not None and (not isinstance(body, dict) or not set(body) <= {"device"}):
        raise bad_request("Expected {device?}")
    try:
        launcher.restart_run(library, record, run, (body or {}).get("device", "auto"), hub.publish_threadsafe)
    finally:
        hub.notify_changed(record.id)
    return Response(status_code=HTTPStatus.ACCEPTED)
