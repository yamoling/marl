"""HTTP routes of MARL Studio (all under `/api`) and shared dependencies."""

from typing import Annotated, Any

from fastapi import Body, Depends, Request

from ..data.library import Library
from ..data.records import ExperimentRecord, RunRecord
from ..errors import bad_request, not_found
from ..security import safe_experiment_path
from ..services.events import EventHub


def get_library(request: Request) -> Library:
    return request.app.state.library


def get_hub(request: Request) -> EventHub:
    return request.app.state.events


LibraryDep = Annotated[Library, Depends(get_library)]
HubDep = Annotated[EventHub, Depends(get_hub)]
JsonBody = Annotated[Any, Body()]
"""Any JSON body (validated by the handlers, which answer 400 with a precise message)."""


def experiment_record(library: Library, experiment_id: str) -> ExperimentRecord:
    """Validated record of an experiment id: 400/403 for unsafe ids, 404 if unknown. @ai-generated"""
    path = safe_experiment_path(library.root, experiment_id)
    record = library.get(path.relative_to(library.root).as_posix())
    if record is None:
        raise not_found(f"Unknown experiment {experiment_id}", "unknown-experiment")
    return record


def run_record(library: Library, run_id: str) -> tuple[ExperimentRecord, RunRecord]:
    """Validated experiment and run of a run id `<experiment id>/<dirname>`. @ai-generated"""
    experiment_id, sep, dirname = run_id.rstrip("/").rpartition("/")
    if not sep or not experiment_id or not dirname:
        raise bad_request("Invalid run id")
    record = experiment_record(library, experiment_id)
    run = record.run(dirname)
    if run is None:
        raise not_found(f"Unknown run {run_id}", "unknown-run")
    return record, run
