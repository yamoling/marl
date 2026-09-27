from http import HTTPStatus
from pathlib import Path

import orjson
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from . import safe_experiment, safe_log_path, safe_run, safe_run_process, state, stop_verified_run

router = APIRouter()


@router.get("/runs/get/{logdir:path}")
def list_runs(logdir: str):
    """List runs from an experiment inside the logs root.

    @ai-edited
    """
    logdir = safe_log_path(logdir)
    exp = safe_experiment(logdir)
    runs = []
    for run in exp.runs:
        safe_run(run, logdir)
        pid = safe_run_process(run)
        if pid is not None:
            status = "RUNNING"
        elif run.is_complete:
            status = "COMPLETED"
        elif run.progress == 0:
            status = "CREATED"
        else:
            status = "CANCELLED"
        runs.append(
            {
                "rundir": run.rundir,
                "seed": run.seed,
                "progress": run.progress,
                "pid": pid.pid if pid is not None else None,
                "status": status,
                "n_tests": run.n_tests,
            }
        )
    return Response(orjson.dumps(runs), media_type="application/json")


@router.post("/runs/stop/{rundir:path}")
def stop_run(rundir: str):
    """Stop a run inside the logs root.

    @ai-edited
    """
    rundir = safe_log_path(rundir)
    logdir = str(Path(rundir).parent)
    exp = safe_experiment(logdir)
    for run in exp.runs:
        if safe_run(run, logdir) == rundir:
            stop_verified_run(run)
            return Response(status_code=HTTPStatus.NO_CONTENT)
    raise HTTPException(HTTPStatus.NOT_FOUND, "Run not found")


@router.post("/runs/start/{rundir:path}")
async def start_run(rundir: str, request: Request):
    """Start a run inside the logs root.

    @ai-edited
    """
    rundir = safe_log_path(rundir)
    logdir = str(Path(rundir).parent)
    exp = safe_experiment(logdir)
    try:
        data = await request.json()
    except ValueError as exc:
        raise HTTPException(HTTPStatus.BAD_REQUEST, "Invalid JSON body") from exc
    if data is not None and not isinstance(data, dict):
        raise HTTPException(HTTPStatus.BAD_REQUEST, "Expected a JSON object")
    device = data.get("device", "auto") if data else "auto"
    for run in exp.runs:
        if safe_run(run, logdir) != rundir:
            continue
        if safe_run_process(run) is None and not run.is_complete:
            try:
                state.new_runs(
                    logdir,
                    n_runs=1,
                    n_tests=1,
                    seed=run.seed,
                    test_interval=run.test_interval,
                    n_jobs=1,
                    device=device,
                    save_weights=run.save_weights,
                    save_actions=run.save_actions,
                )
            except ValueError as exc:
                raise HTTPException(HTTPStatus.BAD_REQUEST, str(exc)) from exc
            except FileNotFoundError as exc:
                raise HTTPException(HTTPStatus.NOT_FOUND, str(exc)) from exc
        return Response(status_code=HTTPStatus.NO_CONTENT)
    raise HTTPException(HTTPStatus.NOT_FOUND, "Run not found")
