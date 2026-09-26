"""Experiment routes: listing, detail, catalog, health, preview, episodes, launching and management."""

from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Query, Response

from ..errors import bad_request, json_response
from ..services import launcher, runs
from . import HubDep, JsonBody, LibraryDep, experiment_record

router = APIRouter(prefix="/api")


@router.get("/experiments")
def list_experiments(
    library: LibraryDep,
    q: str = "",
    algo: str | None = None,
    status: str | None = None,
    health: str | None = None,
):
    """`ExperimentSummary[]` sorted by creation date (most recent first). @ai-generated"""
    return json_response(library.list_summaries(q=q, algo=algo, status=status, health=health))


@router.get("/params")
def params(library: LibraryDep, ids: str = ""):
    """Flattened parameters per experiment; unknown ids are omitted. @ai-generated"""
    return json_response(library.params([i for i in ids.split(",") if i]))


# Routes with a suffix are declared before the bare `{id:path}` routes, which would match them otherwise.


@router.post("/experiments/{experiment_id:path}/health")
def health(experiment_id: str, library: LibraryDep):
    """@ai-generated"""
    record = experiment_record(library, experiment_id)
    return json_response(library.health(record.id))


@router.get("/experiments/{experiment_id:path}/catalog")
def catalog(experiment_id: str, library: LibraryDep):
    """@ai-generated"""
    record = experiment_record(library, experiment_id)
    return json_response(library.catalog(record.id))


@router.get("/experiments/{experiment_id:path}/preview")
def preview(experiment_id: str, library: LibraryDep, points: Annotated[int, Query(ge=4, le=1000)] = 60):
    """@ai-generated"""
    record = experiment_record(library, experiment_id)
    return json_response(library.preview(record.id, points))


@router.get("/experiments/{experiment_id:path}/test-steps")
def test_steps(experiment_id: str, library: LibraryDep):
    """@ai-generated"""
    record = experiment_record(library, experiment_id)
    return json_response(library.test_steps(record.id))


@router.get("/experiments/{experiment_id:path}/episodes")
def episodes(experiment_id: str, step: int, library: LibraryDep):
    """@ai-generated"""
    record = experiment_record(library, experiment_id)
    return json_response(library.episodes(record.id, step))


@router.get("/experiments/{experiment_id:path}/launch-defaults")
def launch_defaults(experiment_id: str, library: LibraryDep):
    """@ai-generated"""
    record = experiment_record(library, experiment_id)
    return json_response(launcher.launch_defaults(library, record))


@router.post("/experiments/{experiment_id:path}/runs")
def start_runs(experiment_id: str, library: LibraryDep, hub: HubDep, body: JsonBody = None):
    """Launch new runs: 202 `{runs}`, 400 invalid, 409 not launchable or seed collision, 502 early failure. @ai-generated"""
    record = experiment_record(library, experiment_id)
    try:
        run_ids = launcher.start_runs(library, record, body, hub.publish_threadsafe)
    finally:
        hub.notify_changed(record.id)
    return json_response({"runs": run_ids}, HTTPStatus.ACCEPTED)


@router.post("/experiments/{experiment_id:path}/stop")
def stop_experiment(experiment_id: str, library: LibraryDep, hub: HubDep):
    """@ai-generated"""
    record = experiment_record(library, experiment_id)
    runs.stop_experiment(library, record)
    hub.notify_changed(record.id)
    return Response(status_code=HTTPStatus.NO_CONTENT)


@router.get("/experiments/{experiment_id:path}")
def detail(experiment_id: str, library: LibraryDep):
    """`ExperimentDetail`. @ai-generated"""
    record = experiment_record(library, experiment_id)
    return json_response(library.detail(record.id))


@router.patch("/experiments/{experiment_id:path}")
def rename(experiment_id: str, library: LibraryDep, hub: HubDep, body: JsonBody = None):
    """Rename (move) an experiment: 200 `{id}`, 409 if the destination exists or runs are active. @ai-generated"""
    record = experiment_record(library, experiment_id)
    if not isinstance(body, dict) or set(body) != {"new_id"}:
        raise bad_request("Expected {new_id}")
    new_id = runs.rename(library, record, body["new_id"])
    hub.notify_changed(new_id)
    return json_response({"id": new_id})


@router.delete("/experiments/{experiment_id:path}")
def delete(experiment_id: str, library: LibraryDep):
    """@ai-generated"""
    record = experiment_record(library, experiment_id)
    runs.delete(library, record)
    return Response(status_code=HTTPStatus.NO_CONTENT)
