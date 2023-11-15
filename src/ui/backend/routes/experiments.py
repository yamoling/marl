from . import app, state
from http import HTTPStatus
from marl import Experiment
from marl.utils.exceptions import ExperimentVersionMismatch


@app.route("/experiment/replay/<path:path>")
def get_episode(path: str):
    return state.replay_episode(path).to_json()


@app.route("/experiment/list")
def list_experiments():
    try:
        return state.list_experiments()
    except ExperimentVersionMismatch as e:
        print(e)
        return str(e), HTTPStatus.INTERNAL_SERVER_ERROR


@app.route("/experiment/<path:logdir>", methods=["GET"])
def get_experiment(logdir: str):
    return Experiment.get_parameters(logdir)
