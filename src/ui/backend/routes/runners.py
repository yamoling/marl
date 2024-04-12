from . import app
from . import state
from http import HTTPStatus
from flask import request


@app.route("/runner/port/<path:rundir>", methods=["GET"])
def get_runner_port(rundir: str):
    port = state.get_runner_port(rundir)
    if port is None:
        return "", HTTPStatus.NOT_FOUND
    return str(port)


@app.route("/runner/new/<path:logdir>", methods=["POST"])
def new_run(logdir: str):
    data = request.json
    if data is None:
        return ("", HTTPStatus.BAD_REQUEST)
    print(data)
    state.new_runs(logdir, data["nRuns"], data["nTests"], data["seed"])
    return ""
