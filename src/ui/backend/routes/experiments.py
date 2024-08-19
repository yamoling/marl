from . import app, state
from flask import request
from http import HTTPStatus
from serde.json import to_json
import json
import cv2
from marl.exceptions import ExperimentVersionMismatch
from marl.utils import encode_b64_image


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
    exp = state.get_experiment(logdir)
    return to_json(exp)


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


@app.route("/experiment/rename", methods=["POST"])
def rename_experiment():
    json_data = request.json
    if json_data is None:
        return ("", HTTPStatus.BAD_REQUEST)
    logdir = json_data["logdir"]
    new_logdir = json_data["newLogdir"]
    exp = state.get_experiment(logdir)
    exp.move(new_logdir)
    # exp.copy(new_logdir, copy_runs=True)
    state.unload_experiment(logdir)
    state.load_experiment(new_logdir)
    # exp.delete()
    return ("", HTTPStatus.NO_CONTENT)


@app.route("/experiment/delete/<path:logdir>", methods=["DELETE"])
def delete_experiment(logdir: str):
    try:
        exp = state.get_experiment(logdir)
        exp.delete()
        state.unload_experiment(logdir)
    except FileNotFoundError as e:
        return str(e), HTTPStatus.NOT_FOUND
    except AttributeError:  # From version mismatch, for instance
        import shutil

        shutil.rmtree(logdir)
    return ("", HTTPStatus.NO_CONTENT)


@app.route("/experiment/image/<seed>/<path:logdir>")
def get_env_image(seed: str, logdir: str):
    exp = state.get_experiment(logdir)
    exp.env.seed(int(seed))
    exp.env.reset()
    image = exp.env.render(mode="rgb_array")
    image = cv2.resize(image, (100, 100))
    return encode_b64_image(image)


@app.route("/experiment/test-on-other-env", methods=["POST"])
def test_on_other_env():
    json_data = request.json
    if json_data is None:
        return ("", HTTPStatus.BAD_REQUEST)
    logdir = json_data["logdir"]
    new_logdir = json_data["newLogdir"]
    env_logdir = json_data["envLogdir"]
    n_tests = json_data["nTests"]
    exp = state.get_experiment(logdir)
    test_env = state.get_experiment(env_logdir).test_env

    import threading

    def start():
        exp.test_on_other_env(test_env, new_logdir, n_tests, quiet=True)

    threading.Thread(target=start).start()

    # The parent just returns
    return ""
