from http import HTTPStatus

from fastapi import APIRouter
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
                    "pid": run.get_pid(),
                    "status": status,
                }
            )
        return Response(orjson.dumps(runs), media_type="application/json")
    except (ModuleNotFoundError, AttributeError) as e:
        # This can occur if the structure of the repository has changed since the
        # pickle file has been created (ModuleNotFoundError) or if the attributes of the
        # experiment have changed (AttributeError). In both cases, we consider that the experiment has no runs to display, and we return an empty list.
        print(e)
        return Response(orjson.dumps([]), media_type="application/json")


@router.post("/runs/stop/{rundir:path}")
def stop_run(rundir: str):
    try:
        state.stop_run(rundir)
        return Response(status_code=HTTPStatus.NO_CONTENT)
    except FileNotFoundError as e:
        return Response(content=str(e), status_code=HTTPStatus.NOT_FOUND)


@router.post("/runs/start/{rundir:path}")
def start_run(rundir: str):
    try:
        state.start_run(rundir)
        return Response(status_code=HTTPStatus.NO_CONTENT)
    except FileNotFoundError as e:
        return Response(content=str(e), status_code=HTTPStatus.NOT_FOUND)
