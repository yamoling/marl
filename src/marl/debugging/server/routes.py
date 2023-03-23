from http import HTTPStatus
import os
from flask import request
from marl.utils import alpha_num_order
from .messages import MemoryConfig, TrainConfig, StartTrain, GeneratorConfig
from marl.debugging.server import app, replay_state, train_state, state
from . import replay


@app.route("/replay/episode/<path:path>")
def get_episode(path: str):
    # Security issue here !
    return replay.get_episode(path).to_json()

@app.route("/replay/test/list/<path:directory>", methods=["GET"])
def list_test_episodes(directory: str):
    return [e.to_json() for e in replay.get_tests_at(directory)]

@app.route("/experiment/checkpoint/load", methods=["POST"])
def load_checkpoint():
    data: dict = request.get_json()
    state.get_runner(data["logdir"], data.get("checkpoint_dir", None))
    return ""

@app.route("/runner/train/start/<path:logdir>", methods=["POST"])
def start_train(logdir: str):
    data: dict = request.get_json()
    return { "port" : state.train(logdir, StartTrain(**data)) }

@app.route("/experiment/create", methods=["POST"])
def create_experiment():
    data: dict = request.get_json()
    data["level"] = data["level"]
    data["memory"] = MemoryConfig(**data["memory"])
    data["generator"] = GeneratorConfig(**data["generator"])
    train_config = TrainConfig(**data)
    exp = state.create_experiment(train_config)
    return exp.summary
    

@app.route("/experiment/list")
def list_experiments():
    return {logdir: e.summary for logdir, e in state.list_experiments().items()}

@app.route("/experiment/load/<path:logdir>", methods=["GET"])
def load_experiment(logdir: str):
    experiment = state.load_experiment(logdir)
    return {
        **experiment.summary,
        "train": [t.to_json() for t in experiment.train_summary()],
        "test": [t.to_json() for t in experiment.test_summary()],
    }

@app.route("/experiment/load/<path:logdir>", methods=["DELETE"])
def unload_experiment(logdir: str):
    state.stop_experiment(logdir)
    return ""

@app.route("/experiment/delete/<path:logdir>", methods=["DELETE"])
def delete_experiment(logdir: str):
    try:
        state.delete_experiment(logdir)
        return ""
    except ValueError as e:
        return str(e), HTTPStatus.BAD_REQUEST

@app.route("/env/maps/list")
def list_maps():
    return sorted(os.listdir("maps"), key=alpha_num_order)

@app.route("/train/create", methods=["POST"])
def create_algo():
    data: dict = request.get_json()
    data["level"] = data["level"]
    data["memory"] = MemoryConfig(**data["memory"])
    data["generator"] = GeneratorConfig(**data["generator"])
    train_config = TrainConfig(**data)
    port = train_state.create_runner(train_config)
    replay_state.update(train_config.logdir)
    return {
        "logdir": train_config.logdir,
        "port": port
    }


@app.route("/train/start", methods=["POST"])
def train_start():
    data: dict = request.get_json()
    train_state.train(StartTrain(**data))
    return ""


@app.route("/train/memory/priorities")
def get_priorities():
    # For some reason, pylint thinks that the return type cannot be unpacked
    # pylint: disable=E0633
    cumsum, priorities = train_state.get_memory_priorities()
    return { "cumsum": cumsum, "priorities": priorities }

@app.route("/train/memory/transition/<transition_num>")
def get_transition(transition_num: str):
    transition_num = int(transition_num)
    return train_state.get_transition_from_memory(transition_num)
