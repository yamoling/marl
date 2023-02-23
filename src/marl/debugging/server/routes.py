from typing import Literal
import json
import os
from flask import request
from marl.utils import alpha_num_order
from .replay import TestItem
from .train import ALGORITHMS, ENV_WRAPPERS
from marl.debugging.server import app, replay_state, train_state


@app.route("/list/<kind>")
def list_files(kind: Literal["train", "test"]):
    files = replay_state.get_files(kind, with_path=False)
    if kind == "test":
        res = []
        for i, file in enumerate(files):
            episodes = replay_state.get_test_episodes(i)
            with open(os.path.join(replay_state.test_dir, file, "metrics.json"), "r", encoding="utf-8") as f:
                metrics = json.load(f)
            res.append(TestItem(file, episodes, metrics))
        files = res
    return files

@app.route("/episode/train/<episode_number>")
def get_train_episode(episode_num: str):
    file = replay_state.get_episode_file("train", int(episode_num))
    return upload_file(file)

@app.route("/episode/test/<step_number>/<episode_number>")
def get_test_episode(episode_number: str, step_number: str):
    episode_number = int(episode_number)
    step_number = int(step_number)
    file = replay_state.get_episode_file("test", int(episode_number), int(step_number))
    file = os.path.join(replay_state.tests_folders_paths[step_number], file)
    contents = upload_file(file)
    data = json.loads(contents)
    return data


@app.route("/metrics/test/<step_num>/<episode_num>")
def get_test_metrics(step_num, episode_num):
    step_num = int(step_num)
    episode_num = int(episode_num)
    file = replay_state.get_episode_file("test", episode_num, step_num)
    file = os.path.join(replay_state.tests_folders_paths[step_num], file)
    print(file)
    with open(file, "r", encoding="utf-8") as f:
        episode = json.load(f)
        return episode["metrics"]


@app.route("/metrics/<kind>")
def get_metrics(kind: Literal["train", "test"]):
    files = replay_state.get_files(kind, with_path=True)
    metrics = []
    if kind == "train":
        for file in files:
            with open(file, "rb") as f:
                data = json.load(f)
                metrics.append(data["metrics"])
    else:
        for folder in files:
            with open(os.path.join(folder, "metrics.json"), "rb") as f:
                data = json.load(f)
                metrics.append({ **data, "score": data["avg_score"]})
    return metrics
            
@app.route("/frames/test/<step_num>/<episode_num>")
def get_frames(step_num: str, episode_num: str):
    frames = replay_state.get_video_frames(int(step_num), int(episode_num))
    return frames


@app.route("/ls/<path:path>")
def ls(path: str):
    files = sorted(os.listdir(path), key=alpha_num_order)
    files = [os.path.join(path, f) for f in files]
    files = [{"path": f, "isDirectory": os.path.isdir(f)} for f in files]
    return files
    

@app.route("/load/<path:path>")
def load_directory(path: str):
    replay_state._replay_dir = path
    replay_state.update()
    return ""

@app.route("/algo/list")
def get_algorithms():
    return ALGORITHMS

@app.route("/env/wrapper/list")
def get_env_wrappers():
    return ENV_WRAPPERS

@app.route("/env/maps/list")
def list_maps():
    return sorted(os.listdir("maps"), key=alpha_num_order)


@app.route("/algo/create", methods=["POST"])
def create_algo():
    data: dict = request.get_json()
    algo_name = data["algo"]
    wrappers = data["wrappers"]
    time_limit = data.get("timeLimit", None)
    level = data["level"]
    level = os.path.join("maps", level)
    pioritized = data["memory"]["prioritized"]
    memory_size = data["memory"]["size"]
    train_state.create_algo(
        algo_name=algo_name,
        map_file=level,
        wrappers=wrappers,
        time_limit=time_limit,
        memory_size=memory_size,
        prioritized=pioritized
    )
    return ""


@app.route("/train/episode/<episode_num>")
def get_train_episode2(episode_num: str):
    episode_num = int(episode_num)
    return train_state.get_train_episode(episode_num)

@app.route("/train/frames/<episode_num>")
def get_train_frames2(episode_num: str):
    episode_num = int(episode_num)
    return train_state.get_train_frames(episode_num)

def upload_file(filename: str) -> bytes:
    with open(filename, "rb") as f:
        return f.read()