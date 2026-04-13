from http import HTTPStatus
import logging

from fastapi import APIRouter, Request
from fastapi.responses import Response
from . import state
import orjson

router = APIRouter()


@router.get("/runs/get/{logdir:path}")
def list_runs(logdir: str):
    try:
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
                }
            )
        return Response(orjson.dumps(runs), media_type="application/json")
    except (ModuleNotFoundError, AttributeError) as e:
        # This can occur if the structure of the repository has changed since the
        # pickle file has been created (ModuleNotFoundError) or if the attributes of the
        # experiment have changed (AttributeError). In both cases, we consider that the experiment has no runs to display, and we return an empty list.
        logging.error(e)
        return Response(orjson.dumps([]), media_type="application/json")


@router.post("/runs/stop/{rundir:path}")
def stop_run(rundir: str):
    try:
        state.stop_run(rundir)
        return Response(status_code=HTTPStatus.NO_CONTENT)
    except FileNotFoundError as e:
        return Response(content=str(e), status_code=HTTPStatus.NOT_FOUND)


@router.post("/runs/start/{rundir:path}")
async def start_run(rundir: str, request: Request):
    try:
        device = "auto"
        try:
            data = await request.json()
            if data is not None and "device" in data:
                device = data["device"]
        except Exception:
            # If no JSON body, default to auto
            pass
        state.start_run(rundir, device=device)
        return Response(status_code=HTTPStatus.NO_CONTENT)
    except FileNotFoundError as e:
        return Response(content=str(e), status_code=HTTPStatus.NOT_FOUND)
