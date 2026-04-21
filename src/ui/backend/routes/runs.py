from http import HTTPStatus

import orjson
from fastapi import APIRouter, Request
from fastapi.responses import Response

from . import state

router = APIRouter()


@router.get("/runs/get/{logdir:path}")
def list_runs(logdir: str):
    exp = state.get_experiment(logdir)
    runs = []
    for run in exp.runs:
        if run.is_running:
            status = "RUNNING"
        elif run.is_completed(exp.n_steps):
            status = "COMPLETED"
        else:
            progress = run.get_progress(exp.n_steps)
            if progress == 0:
                status = "CREATED"
            else:
                status = "CANCELLED"
        runs.append(
            {
                "rundir": run.rundir,
                "seed": run.seed,
                "progress": run.get_progress(exp.n_steps),
                "pid": run.pid,
                "status": status,
                "n_tests": run.n_tests,
            }
        )
    return Response(orjson.dumps(runs), media_type="application/json")


@router.post("/runs/stop/{rundir:path}")
def stop_run(rundir: str):
    state.stop_run(rundir)
    return Response(status_code=HTTPStatus.NO_CONTENT)


@router.post("/runs/start/{rundir:path}")
async def start_run(rundir: str, request: Request):
    device = "auto"
    try:
        data = await request.json()
        if data is not None and "device" in data:
            device = data["device"]
    except Exception:
        pass
    state.start_run(rundir, device=device)
    return Response(status_code=HTTPStatus.NO_CONTENT)
