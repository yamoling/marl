import logging
import shutil
import time
from http import HTTPStatus
from pathlib import Path
from signal import SIGINT
from typing import cast

import orjson
import psutil
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

import marl
from marl.utils import encode_b64_image

from . import safe_experiment, safe_log_path, safe_run, safe_run_process, state, stop_verified_run, verified_launcher

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/experiment/replay/{time_step}/{test_num}/{only_saved_actions}/{rundir:path}")
def replay(time_step: int, test_num: int, only_saved_actions: bool, rundir: str):
    """Replay an episode from a run inside the logs root.

    @ai-edited
    """
    rundir = safe_log_path(rundir)
    logdir = str(Path(rundir).parent)
    exp = safe_experiment(logdir, full=True)
    run = next((run for run in exp.runs if safe_run(run, logdir) == rundir), None)
    if run is None:
        raise HTTPException(HTTPStatus.NOT_FOUND, "Run not found")
    full_run = run.to_full()
    if safe_run(full_run, logdir) != rundir:
        raise HTTPException(HTTPStatus.FORBIDDEN, "Run metadata points to another directory")
    replay_episode = full_run.replay_episode(time_step, test_num, only_saved_actions=only_saved_actions)
    serialized = orjson.dumps(replay_episode, option=orjson.OPT_SERIALIZE_NUMPY, default=marl.utils.default_serialization)
    return Response(serialized, media_type="application/json", status_code=HTTPStatus.OK)


@router.get("/experiment/list")
def list_experiments():
    """List experiments using the configured logs directory.

    @ai-edited
    """
    return state.list_experiments()


@router.get("/experiment/is_running/{logdir:path}")
def list_running_experiments(logdir: str):
    """Read the running status of a local experiment.

    @ai-edited
    """
    logdir = safe_log_path(logdir)
    exp = safe_experiment(logdir)
    runs = list(exp.runs)
    for run in runs:
        safe_run(run, logdir)
    running = any(safe_run_process(run) is not None for run in runs)
    return Response(orjson.dumps(running), media_type="application/json")


@router.post("/experiment/load/{logdir:path}")
def load_experiment(logdir: str):
    """
    Load an experiment into the state.
    This does not return anything but make the backend gain time if the user wants to
    replay an episode in the future.

    @ai-edited
    """
    logdir = safe_log_path(logdir)
    state.load_experiment(logdir)
    return Response(status_code=HTTPStatus.NO_CONTENT)


@router.delete("/experiment/load/{logdir:path}")
def unload_experiment(logdir: str):
    """Unload only an experiment under the logs root.

    @ai-edited
    """
    logdir = safe_log_path(logdir)
    state.unload_experiment(logdir)
    return Response(status_code=HTTPStatus.NO_CONTENT)


@router.post("/experiment/rename")
async def rename_experiment(request: Request):
    """Move an experiment only between distinct paths in the logs root.

    @ai-edited
    """
    try:
        json_data = await request.json()
    except ValueError as exc:
        raise HTTPException(HTTPStatus.BAD_REQUEST, "Invalid JSON body") from exc
    if not isinstance(json_data, dict) or not isinstance(json_data.get("logdir"), str) or not isinstance(json_data.get("newLogdir"), str):
        raise HTTPException(HTTPStatus.BAD_REQUEST, "Expected logdir and newLogdir strings")
    logdir = safe_log_path(json_data["logdir"])
    new_logdir = safe_log_path(json_data["newLogdir"], must_exist=False)
    if Path(new_logdir).exists() or Path(new_logdir).is_symlink():
        raise HTTPException(HTTPStatus.CONFLICT, "Destination already exists")
    if Path(new_logdir).is_relative_to(Path(logdir)):
        raise HTTPException(HTTPStatus.BAD_REQUEST, "Cannot move an experiment into itself")
    if not Path(new_logdir).parent.is_dir():
        raise HTTPException(HTTPStatus.NOT_FOUND, "Destination parent does not exist")
    exp = safe_experiment(logdir)
    for run in exp.runs:
        safe_run(run, logdir)
        if safe_run_process(run) is not None:
            raise HTTPException(HTTPStatus.CONFLICT, "Cannot rename an active experiment")
    exp.move(Path(new_logdir))
    state.unload_experiment(logdir)
    state.load_experiment(new_logdir)
    return Response(status_code=HTTPStatus.NO_CONTENT)


@router.delete("/experiment/delete/{logdir:path}")
def delete_experiment(logdir: str):
    """Delete only the validated experiment directory, including on legacy fallback.

    @ai-edited
    """
    logdir = safe_log_path(logdir)
    try:
        exp = safe_experiment(logdir)
    except AttributeError:  # Legacy version mismatch while loading the experiment.
        shutil.rmtree(safe_log_path(logdir))
    else:
        for run in exp.runs:
            safe_run(run, logdir)
            if safe_run_process(run) is not None:
                raise HTTPException(HTTPStatus.CONFLICT, "Cannot delete an active experiment")
        shutil.rmtree(safe_log_path(logdir))
        state.unload_experiment(logdir)
    return Response(status_code=HTTPStatus.NO_CONTENT)


@router.post("/experiment/stop-runs/{logdir:path}")
def stop_experiment_runs(logdir: str):
    """Kill local experiment runs, including queued runs that start after the current ones.

    @ai-edited
    """
    logdir = safe_log_path(logdir)
    exp = safe_experiment(logdir)
    while True:
        runs = list(exp.runs)
        for run in runs:
            safe_run(run, logdir)
        active = [(run, process) for run in runs if (process := safe_run_process(run)) is not None]
        if not active:
            break
        launchers = {}
        for _, process in active:
            launcher = verified_launcher(process, Path(logdir))
            if launcher is not None and launcher.pid != process.pid:
                launchers[launcher.pid] = launcher
        for run, _ in active:
            stop_verified_run(run)
        for launcher in launchers.values():
            try:
                launcher.send_signal(SIGINT)
            except psutil.NoSuchProcess:
                pass
        time.sleep(1)
    return Response(status_code=HTTPStatus.NO_CONTENT)


@router.get("/experiment/image/{seed}/{logdir:path}")
def get_env_image(seed: int, logdir: str):
    """Render an environment belonging to a local experiment.

    @ai-edited
    """
    logdir = safe_log_path(logdir)
    exp = cast(marl.Experiment, safe_experiment(logdir, full=True))
    env = exp.env.make()
    env.reset(seed=seed)
    return encode_b64_image(env.get_image())


@router.get("/experiment/{logdir:path}")
def get_experiment(logdir: str):
    """Read an experiment from the configured logs root.

    @ai-edited
    """
    logdir = safe_log_path(logdir)
    return Response(marl.Experiment.load(logdir).to_json(), media_type="application/json")
