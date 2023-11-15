from .. import app
from . import state
from ..messages import RunConfig, TrainConfig
from flask import request
from http import HTTPStatus


@app.route("/runner/create/<path:logdir>", methods=["POST"])
def create_runner(logdir: str):
    run_config = RunConfig(**request.get_json())
    state.create_runner(logdir, run_config)
    return ""


@app.route("/runner/port/<path:rundir>", methods=["GET"])
def get_runner_port(rundir: str):
    port = state.get_runner_port(rundir)
    if port is None:
        return "", HTTPStatus.NOT_FOUND
    return str(port)


@app.route("/runner/restart/<path:rundir>", methods=["POST"])
def restart_runner(rundir: str):
    data = TrainConfig(**request.get_json())
    state.restart_runner(rundir, data)
    return ""


@app.route("/runner/stop/<path:rundir>", methods=["POST"])
def stop_runner(rundir: str):
    state.stop_runner(rundir)
    return ""


@app.route("/runner/delete/<path:rundir>", methods=["DELETE"])
def delete_runner(rundir: str):
    state.delete_runner(rundir)
    return ""
