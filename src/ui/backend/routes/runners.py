from . import app
from . import state
from http import HTTPStatus


@app.route("/runner/port/<path:rundir>", methods=["GET"])
def get_runner_port(rundir: str):
    port = state.get_runner_port(rundir)
    if port is None:
        return "", HTTPStatus.NOT_FOUND
    return str(port)
