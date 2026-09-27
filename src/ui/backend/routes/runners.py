from http import HTTPStatus

from fastapi import APIRouter, Request
from fastapi.responses import Response

from . import state

router = APIRouter()


@router.post("/runner/new/{logdir:path}")
async def new_run(logdir: str, request: Request):
    """Reject invalid launches and surface immediate child failures to the client. @ai-edited"""
    try:
        data = await request.json()
    except ValueError:
        return Response(content="Invalid JSON body", status_code=HTTPStatus.BAD_REQUEST)
    if not isinstance(data, dict):
        return Response(content="Expected a JSON object", status_code=HTTPStatus.BAD_REQUEST)
    required_keys = ("nRuns", "nTests", "seed")
    if not all(key in data for key in required_keys):
        return Response(content="Missing nRuns, nTests or seed", status_code=HTTPStatus.BAD_REQUEST)
    try:
        state.new_runs(
            logdir,
            data["nRuns"],
            data["nTests"],
            data["seed"],
            test_interval=data.get("testInterval", 5000),
            n_jobs=data.get("nJobs", 1),
            device=data.get("device", "auto"),
            gpu_strategy=data.get("gpuStrategy", "group"),
            disabled_devices=data.get("disabledDevices"),
            save_weights=data.get("saveWeights", False),
            save_actions=data.get("saveActions", True),
        )
    except ValueError as exc:
        return Response(content=str(exc), status_code=HTTPStatus.BAD_REQUEST)
    except FileNotFoundError as exc:
        return Response(content=str(exc), status_code=HTTPStatus.NOT_FOUND)
    except (OSError, RuntimeError) as exc:
        return Response(content=str(exc), status_code=HTTPStatus.BAD_GATEWAY)
    return Response(content="", status_code=HTTPStatus.OK)
