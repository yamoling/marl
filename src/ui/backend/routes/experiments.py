import logging
import time
from http import HTTPStatus

import cv2
import orjson
from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse, Response

import marl
from marl.exceptions import ExperimentVersionMismatch
from marl.utils import encode_b64_image

from . import state

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/experiment/replay/{path:path}")
def replay(path: str):
    try:
        replay_episode = state.replay_episode(path)
        serialized = orjson.dumps(replay_episode, option=orjson.OPT_SERIALIZE_NUMPY, default=marl.utils.default_serialization)
        return Response(serialized, media_type="application/json", status_code=HTTPStatus.OK)
    except ValueError as e:
        return PlainTextResponse(str(e), status_code=HTTPStatus.INTERNAL_SERVER_ERROR)


@router.get("/experiment/list")
def list_experiments():
    try:
        return state.list_experiments()
    except ExperimentVersionMismatch as e:
        logger.exception("Failed to list experiments due to version mismatch")
        return PlainTextResponse(str(e), status_code=HTTPStatus.INTERNAL_SERVER_ERROR)


@router.get("/experiment/is_running/{logdir:path}")
def list_running_experiments(logdir: str):
    try:
        exp = state.get_experiment(logdir)
        return Response(orjson.dumps(exp.is_running), media_type="application/json")
    except (ModuleNotFoundError, AttributeError):
        return Response(orjson.dumps(False), media_type="application/json")


@router.get("/experiment/{logdir:path}")
def get_experiment(logdir: str):
    return Response(orjson.dumps(marl.Experiment.get_parameters(logdir)), media_type="application/json")


@router.post("/experiment/load/{logdir:path}")
def load_experiment(logdir: str):
    """
    Load an experiment into the state.
    This does not return anything but make the backend gain time if the user wants to
    replay an episode in the future.
    """
    state.load_experiment(logdir)
    return Response(status_code=HTTPStatus.NO_CONTENT)


@router.delete("/experiment/load/{logdir:path}")
def unload_experiment(logdir: str):
    state.unload_experiment(logdir)
    return Response(status_code=HTTPStatus.NO_CONTENT)


@router.post("/experiment/rename")
async def rename_experiment(request: Request):
    json_data = await request.json()
    if json_data is None:
        return Response(status_code=HTTPStatus.BAD_REQUEST)
    logdir = json_data["logdir"]
    new_logdir = json_data["newLogdir"]
    exp = state.get_experiment(logdir)
    exp.move(new_logdir)
    # exp.copy(new_logdir, copy_runs=True)
    state.unload_experiment(logdir)
    state.load_experiment(new_logdir)
    # exp.delete()
    return Response(status_code=HTTPStatus.NO_CONTENT)


@router.delete("/experiment/delete/{logdir:path}")
def delete_experiment(logdir: str):
    try:
        exp = state.get_experiment(logdir)
        exp.delete()
        state.unload_experiment(logdir)
    except FileNotFoundError as e:
        return PlainTextResponse(str(e), status_code=HTTPStatus.NOT_FOUND)
    except AttributeError:  # From version mismatch, for instance
        import shutil

        shutil.rmtree(logdir)
    return Response(status_code=HTTPStatus.NO_CONTENT)


@router.post("/experiment/stop-runs/{logdir:path}")
def stop_experiment_runs(logdir: str):
    """Kill all running runs of an experiment. The loop accounts for queued runs that would start after killing the current ones."""
    try:
        exp = state.get_experiment(logdir)
        while exp.is_running:
            exp.kill_runs()
            time.sleep(0.1)
        return Response(status_code=HTTPStatus.NO_CONTENT)
    except FileNotFoundError as e:
        return PlainTextResponse(str(e), status_code=HTTPStatus.NOT_FOUND)


@router.get("/experiment/image/{seed}/{logdir:path}")
def get_env_image(seed: str, logdir: str):
    exp = state.get_experiment(logdir)
    exp.env.seed(int(seed))
    exp.env.reset()
    image = exp.env.get_image()
    image = cv2.resize(image, (100, 100))
    return encode_b64_image(image)
