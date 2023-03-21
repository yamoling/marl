import os
from flask import request
from marl.utils import alpha_num_order
from .messages import MemoryConfig, TrainConfig, StartTrain, GeneratorConfig
from marl.debugging.server import app, replay_state, train_state, state


@app.route("/replay/episode/<path:path>")
def get_episode(path: str):
    # Security issue here !
    return replay_state.get_episode(path).to_json()

@app.route("/replay/tests/summary/<path:path>")
def get_test_summary(path: str):
    """Load the test episodes from a test directory"""
    return [e.to_json() for e in replay_state.get_tests_at(path)]

@app.route("/replay/checkpoint/load/<path:path>")
def load_checkpoint(path: str):
    train_state.load_checkpoint(path)
    return ""

@app.route("/experiment/list")
def list_experiments():
    return {logdir: e.summary for logdir, e in state.list_experiments().items()}

@app.route("/experiment/delete/<path:logdir>", methods=["DELETE"])
def delete_experiment(logdir: str):
    try:
        state.delete_experiment(logdir)
        return ""
    except ValueError as e:
        return str(e), 400

@app.route("/ls/<path:path>")
def ls(path: str):
    files = sorted(os.listdir(path), key=alpha_num_order)
    files = [os.path.join(path, f) for f in files]
    files = [{"path": f, "isDirectory": os.path.isdir(f)} for f in files]
    return files
    

@app.route("/load/<path:path>")
def load_directory(path: str):
    print("Load directory", path)
    replay_state.update(path)
    train, test = replay_state.experiment_summary()
    train = [t.to_json() for t in train]
    test = [t.to_json() for t in test]
    return {
        "train": train,
        "test": test,
        "envInfo": replay_state.env_info(),
        "algoInfo" : {},
        "timestamp": 0
    }

@app.route("/env/maps/list")
def list_maps():
    return sorted(os.listdir("maps"), key=alpha_num_order)


@app.route("/train/create", methods=["POST"])
def create_algo():
    print("Create algo")
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


def upload_file(filename: str) -> bytes:
    with open(filename, "rb") as f:
        return f.read()