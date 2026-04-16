from http import HTTPStatus
from fastapi import APIRouter, Request
from fastapi.responses import Response
from . import state

router = APIRouter()


@router.get("/runner/port/{rundir:path}")
def get_runner_port(rundir: str):
    port = state.get_runner_port(rundir)
    if port is None:
        return Response(content="", status_code=HTTPStatus.NOT_FOUND)
    return Response(content=str(port))


@router.post("/runner/new/{logdir:path}")
async def new_run(logdir: str, request: Request):
    data = await request.json()
    if data is None:
        return Response(content="", status_code=HTTPStatus.BAD_REQUEST)
    if not all(key in data for key in ("nRuns", "nTests", "seed")):
        return Response(content="Missing nRuns, nTests or seed", status_code=HTTPStatus.BAD_REQUEST)
    gpu_strategy = data.get("gpuStrategy", "group")
    disabled_devices = data.get("disabledDevices", [])
    state.new_runs(
        logdir,
        data["nRuns"],
        data["nTests"],
        data["seed"],
        gpu_strategy=gpu_strategy,
        disabled_devices=disabled_devices,
    )
    return Response(content="")
