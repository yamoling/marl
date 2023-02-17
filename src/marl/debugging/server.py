from dataclasses import dataclass
import os
import json
import base64
import cv2
from typing import Literal
from flask import Flask, request
from flask_cors import CORS
from rlenv.models import Metrics
from marl.utils import alpha_num_order
import marl
import rlenv
from laser_env import LaserEnv

app = Flask(__name__)
CORS(app)
ALGORITHMS = [ "DQN", "RDQN", "VDN linear", "VDN recurrent"]
ENV_WRAPPERS = ["TimeLimit", "VideoRecorder", "ExtrinsicReward", "AgentId"]

@dataclass
class TestItem:
    filename: str
    episodes: list[str]
    metrics: list[Metrics]

@dataclass
class ServerState:
    train_episodes_paths: list[str]
    tests_folders_paths: list[str]
    runner: marl.debugging.DebugRunner
    _replay_dir: str = None

    def __init__(self, replay_dir: str=None) -> None:
        self._replay_dir = replay_dir
        self.train_episodes_paths = []
        self.tests_folders_paths = []
        self.algo = None
        if replay_dir is not None:
            self.update()

    def create_algo(self, algo_name: str, map_file: str, wrappers: list[str], time_limit: int|None):
        builder = rlenv.Builder(LaserEnv(map_file))
        for wrapper in wrappers:
            match wrapper:
                case "TimeLimit": builder.time_limit(time_limit)
                case "VideoRecorder": builder.record("videos")
                case "ExtrinsicReward": builder.extrinsic_reward("linear", initial_reward=0.5, anneal=10)
                case "AgentId": builder.agent_id()
                case other: raise ValueError(f"Unknown wrapper: {wrapper}")
        env, test_env = builder.build_all()
        logdir = os.path.join("logs", "debug")
        match algo_name:
            case "DQN" | "VDN linear": 
                qnetwork = marl.nn.model_bank.MLP.from_env(env)
                algo = marl.qlearning.DQN(qnetwork=qnetwork)
                if algo_name == "VDN linear":
                    algo = marl.qlearning.vdn.VDN(algo)
            case "RDQN" | "VDN recurrent": 
                qnetwork = marl.nn.model_bank.RNNQMix.from_env(env)
                algo = marl.qlearning.RDQN(qnetwork=qnetwork)
                if algo_name == "VDN recurrent":
                    algo = marl.qlearning.vdn.VDN(algo)
            case other: raise ValueError(f"Unknown algorithm: {algo_name}")
        self.runner = marl.debugging.DebugRunner(env, test_env=test_env, algo=algo, logdir=logdir)

    def set_algo(self, algo: marl.RLAlgo):
        self.algo = algo

    def update(self):
        paths = sorted(os.listdir(self.train_dir), key=alpha_num_order)
        self.train_episodes_paths = [os.path.join(self.train_dir, t) for t in paths]
        paths = sorted(os.listdir(self.test_dir), key=alpha_num_order)
        self.tests_folders_paths = [os.path.join(self.test_dir, t) for t in paths if t.startswith("step")]

    @property
    def train_dir(self) -> str:
        return os.path.join(self._replay_dir, "train/")

    @property
    def test_dir(self) -> str:
        return os.path.join(self._replay_dir, "test/")

    def get_files(self, kind: Literal["train", "test"], with_path=False) -> list[str]:
        match kind:
            case "test": files = state.tests_folders_paths
            case "train":files = state.train_episodes_paths
            case other: raise ValueError()
        if not with_path:
            files = [os.path.basename(f) for f in files]
        return files

    def get_test_episodes(self, step_num: int) -> list[str]:
        files = os.listdir(self.tests_folders_paths[step_num])
        json_files = [f for f in files if f.endswith(".json")]
        episode_files = [f for f in json_files if f.split('.')[0].isnumeric()]
        return sorted(episode_files, key=alpha_num_order)

    def get_episode_file(self, kind: Literal["train", "test"], episode_num: int, step_num:int=None):
        match kind:
            case "train": return self.train_episodes_paths[step_num]
            case "test": return self.get_test_episodes(step_num)[episode_num]
            case other: raise ValueError(f"Invalid kind: {other}")

    def get_video_frames(self, step_num: int, episode_num: int) -> list[str]:
        files = os.listdir(self.tests_folders_paths[step_num])
        files = [f for f in files if f.endswith(".mp4")]
        files = sorted(files, key=alpha_num_order)
        video_file = os.path.join(self.tests_folders_paths[step_num], files[episode_num])
        encoded_frames = []
        cap = cv2.VideoCapture(video_file)
        success, frame = cap.read()
        while success:
            jpg_frame = cv2.imencode(".jpg", frame)[1]
            b64_frame = base64.b64encode(jpg_frame)
            str_b64_frame = b64_frame.decode("ascii")
            encoded_frames.append(str_b64_frame)
            success, frame = cap.read()
        return encoded_frames

state: ServerState = ServerState(None)




@app.route("/list/<kind>")
def list_files(kind: Literal["train", "test"]):
    files = state.get_files(kind, with_path=False)
    if kind == "test":
        res = []
        for i, file in enumerate(files):
            episodes = state.get_test_episodes(i)
            with open(os.path.join(state.test_dir, file, "metrics.json"), "r", encoding="utf-8") as f:
                metrics = json.load(f)
            res.append(TestItem(file, episodes, metrics))
        files = res
    return files

@app.route("/episode/train/<episode_number>")
def get_train_episode(episode_num: str):
    file = state.get_episode_file("train", int(episode_num))
    return upload_file(file)

@app.route("/episode/test/<step_number>/<episode_number>")
def get_test_episode(episode_number: str, step_number: str):
    episode_number = int(episode_number)
    step_number = int(step_number)
    file = state.get_episode_file("test", int(episode_number), int(step_number))
    file = os.path.join(state.tests_folders_paths[step_number], file)
    contents = upload_file(file)
    data = json.loads(contents)
    return data


@app.route("/metrics/test/<step_num>/<episode_num>")
def get_test_metrics(step_num, episode_num):
    step_num = int(step_num)
    episode_num = int(episode_num)
    file = state.get_episode_file("test", episode_num, step_num)
    file = os.path.join(state.tests_folders_paths[step_num], file)
    print(file)
    with open(file, "r", encoding="utf-8") as f:
        episode = json.load(f)
        return episode["metrics"]


@app.route("/metrics/<kind>")
def get_metrics(kind: Literal["train", "test"]):
    files = state.get_files(kind, with_path=True)
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
    frames = state.get_video_frames(int(step_num), int(episode_num))
    return frames


@app.route("/ls/<path:path>")
def ls(path: str):
    files = sorted(os.listdir(path), key=alpha_num_order)
    files = [os.path.join(path, f) for f in files]
    files = [{"path": f, "isDirectory": os.path.isdir(f)} for f in files]
    return files
    

@app.route("/load/<path:path>")
def load_directory(path: str):
    state._replay_dir = path
    state.update()
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
    data = request.get_json()
    algo_name = data["algo"]
    wrappers = data["wrappers"]
    time_limit = data.get("timeLimit", None)
    level = data["level"]
    level = os.path.join("maps", level)
    print(algo_name, wrappers, time_limit, level)
    state.create_algo(algo_name, level, wrappers, time_limit)
    return state.runner.get_state()

@app.route("/algo/train/<n_steps>")
def algo_train(n_steps: str):
    n_steps = int(n_steps)
    state.runner.train(n_steps)
    return state.runner.get_state()

def upload_file(filename: str) -> bytes:
    with open(filename, "rb") as f:
        return f.read()


def run(port=5174, debug=False):
    app.run("0.0.0.0", port=port, debug=debug)