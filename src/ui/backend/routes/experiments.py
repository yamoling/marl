from . import app, state
from http import HTTPStatus
from serde.json import to_json
import json
from marl.models import Experiment
from marl.utils.exceptions import ExperimentVersionMismatch


@app.route("/experiment/replay/<path:path>")
def replay(path: str):
    try:
        return to_json(state.replay_episode(path)), HTTPStatus.OK
    except ValueError as e:
        return str(e), HTTPStatus.INTERNAL_SERVER_ERROR


@app.route("/experiment/list")
def list_experiments():
    try:
        return state.list_experiments()
    except ExperimentVersionMismatch as e:
        print(e)
        return str(e), HTTPStatus.INTERNAL_SERVER_ERROR


@app.route("/experiment/is_running/<path:logdir>")
def list_running_experiments(logdir: str):
    try:
        exp = state.get_experiment(logdir)
        return json.dumps(exp.is_running)
    except (ModuleNotFoundError, AttributeError):
        return json.dumps(False)


@app.route("/experiment/<path:logdir>", methods=["GET"])
def get_experiment(logdir: str):
    return Experiment.get_parameters(logdir)


@app.route("/experiment/load/<path:logdir>", methods=["POST"])
def load_experiment(logdir: str):
    """
    Load an experiment into the state.
    This does not return anything but make the backend gain time if the user wants to
    replay an episode in the future.
    """
    state.load_experiment(logdir)
    return ("", HTTPStatus.NO_CONTENT)


@app.route("/experiment/load/<path:logdir>", methods=["DELETE"])
def unload_experiment(logdir: str):
    state.unload_experiment(logdir)
    return ("", HTTPStatus.NO_CONTENT)
