from http import HTTPStatus
from typing import Any, cast

from fastapi import APIRouter, Request
from fastapi.responses import Response

from . import state

router = APIRouter()


@router.post("/runner/new/{logdir:path}")
async def new_run(logdir: str, request: Request):
    data = cast(dict[str, Any], await request.json())
    if data is None:
        return Response(content="", status_code=HTTPStatus.BAD_REQUEST)
    required_keys = ("nRuns", "nTests", "seed")
    if not all(key in data for key in required_keys):
        return Response(content="Missing nRuns, nTests or seed", status_code=HTTPStatus.BAD_REQUEST)
    state.new_runs(
        logdir,
        data["nRuns"],
        data["nTests"],
        data["seed"],
        test_interval=data.get("testInterval", 5000),
        n_jobs=data.get("nJobs", 1),
        gpu_strategy=data.get("gpuStrategy", "group"),
        disabled_devices=data.get("disabledDevices", None),
        save_weights=data.get("saveWeights", False),
        save_actions=data.get("saveActions", True),
    )
    return Response(content="")
