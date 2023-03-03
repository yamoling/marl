import os
from flask import request
from marl.utils import alpha_num_order
from .replay import Item
from .train import MemoryConfig, TrainConfig
from marl.debugging.server import app, replay_state, train_state


@app.route("/replay/train/list")
def list_train_files():
    return replay_state.get_files("train")

@app.route("/replay/test/list")
def list_test_files():
    files = replay_state.get_files("test")
    res = []
    for i, file in enumerate(files):
        # episodes = replay_state.get_test_episodes(i)
        # with open(os.path.join(replay_state.test_dir, file, "metrics.json"), "r", encoding="utf-8") as f:
        #     metrics = json.load(f)
        res.append(Item(file, {}))
    files = res
    return files


@app.route("/replay/episode/<path:path>")
def get_episode(path: str):
    # Security issue here !
    print(path)
    return replay_state.get_episode(path).to_json()


@app.route("/ls/<path:path>")
def ls(path: str):
    files = sorted(os.listdir(path), key=alpha_num_order)
    files = [os.path.join(path, f) for f in files]
    files = [{"path": f, "isDirectory": os.path.isdir(f)} for f in files]
    return files
    

@app.route("/load/<path:path>")
def load_directory(path: str):
    replay_state.update(path)
    train, test = replay_state.experiment_summary()
    train = [t.to_json() for t in train]
    test = [t.to_json() for t in test]
    return {
        "train": train,
        "test": test
    }



@app.route("/env/maps/list")
def list_maps():
    return sorted(os.listdir("maps"), key=alpha_num_order)


@app.route("/algo/create", methods=["POST"])
def create_algo():
    data: dict = request.get_json()
    data["level"] = os.path.join("maps", data["level"])
    data["memory"] = MemoryConfig(**data["memory"])
    train_config = TrainConfig(**data)
    logdir = train_state.create_algo(train_config)
    replay_state.update(logdir)
    return logdir

@app.route("/train/episode/<episode_num>")
def get_train_episode2(episode_num: str):
    episode_num = int(episode_num)
    return train_state.get_train_episode(episode_num)

@app.route("/train/frames/<episode_num>")
def get_train_frames(episode_num: str):
    episode_num = int(episode_num)
    return train_state.get_train_frames(episode_num)

@app.route("/train/memory/priorities")
def get_priorities():
    cumsum, priorities = train_state.get_memory_priorities()
    return { "cumsum": cumsum, "priorities": priorities }

@app.route("/train/memory/transition/<transition_num>")
def get_transition(transition_num: str):
    transition_num = int(transition_num)
    return train_state.get_transition_from_memory(transition_num)


def upload_file(filename: str) -> bytes:
    with open(filename, "rb") as f:
        return f.read()