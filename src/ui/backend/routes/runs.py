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
            runs.append(
                {
                    "rundir": run.rundir,
                    "seed": run.seed,
                    "progress": run.get_progress(exp.n_steps),
                    "pid": run.get_pid(),
                }
            )
        return Response(orjson.dumps(runs), media_type="application/json")
    except (ModuleNotFoundError, AttributeError) as e:
        # This can occur if the structure of the repository has changed since the
        # pickle file has been created (ModuleNotFoundError) or if the attributes of the
        # experiment have changed (AttributeError). In both cases, we consider that the experiment has no runs to display, and we return an empty list.
        print(e)
        return Response(orjson.dumps([]), media_type="application/json")
