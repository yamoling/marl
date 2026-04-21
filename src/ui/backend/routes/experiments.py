import logging
import shutil
import time
from http import HTTPStatus

import orjson
from fastapi import APIRouter, Request
from fastapi.responses import Response

import marl
from marl.utils import encode_b64_image

from . import state

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/experiment/replay/{time_step}/{test_num}/{rundir:path}")
def replay(time_step: int, test_num: int, rundir: str):
    replay_episode = state.replay_episode(rundir, time_step, test_num)
    serialized = orjson.dumps(replay_episode, option=orjson.OPT_SERIALIZE_NUMPY, default=marl.utils.default_serialization)
    return Response(serialized, media_type="application/json", status_code=HTTPStatus.OK)


@router.get("/experiment/list")
def list_experiments():
    return state.list_experiments()


@router.get("/experiment/is_running/{logdir:path}")
def list_running_experiments(logdir: str):
    exp = state.get_experiment(logdir)
    return Response(orjson.dumps(exp.is_running), media_type="application/json")


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
    except AttributeError:  # From version mismatch, for instance
        shutil.rmtree(logdir)
    return Response(status_code=HTTPStatus.NO_CONTENT)


@router.post("/experiment/stop-runs/{logdir:path}")
def stop_experiment_runs(logdir: str):
    """Kill all running runs of an experiment. The loop accounts for queued runs that would start after killing the current ones."""
    exp = state.get_experiment(logdir)
    while exp.is_running:
        exp.kill_runs()
        time.sleep(1)
    return Response(status_code=HTTPStatus.NO_CONTENT)


@router.get("/experiment/image/{seed}/{logdir:path}")
def get_env_image(seed: str, logdir: str):
    exp = state.get_experiment(logdir)
    exp.env.seed(int(seed))
    exp.env.reset()
    image = exp.env.get_image()
    return encode_b64_image(image)
