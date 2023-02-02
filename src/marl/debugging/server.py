from dataclasses import dataclass
import os
import json
import base64
import cv2
from typing import Literal
from flask import Flask
from flask_cors import CORS
from rlenv.models import Metrics, Episode
from marl.utils import alpha_num_order

app = Flask(__name__)
CORS(app)


@dataclass
class Test:
    name: str
    episodes: list[str]
    metrics: Metrics

    def to_json(self):
        return {
            "name": self.name,
            "episodes": self.episodes
        }


@dataclass
class ServerState:
    train_episodes_paths: list[str]
    tests_folders_paths: list[str]
    _replay_dir: str = None

    def __init__(self, replay_dir: str=None) -> None:
        self._replay_dir = replay_dir
        self.train_episodes_paths = []
        self.tests_folders_paths = []
        self.update()

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

    def get_episode_file(self, kind: Literal["train", "test"], episode_num: int, step_num:int=None):
        match kind:
            case "train": return self.train_episodes_paths[step_num]
            case "test": 
                test_episodes = os.listdir(self.tests_folders_paths[step_num])
                test_episodes = sorted(test_episodes, key=alpha_num_order)
                return os.path.join(self.tests_folders_paths[step_num], test_episodes[episode_num])
            case other: raise ValueError()

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

state = ServerState("files/")




@app.route("/list/<kind>")
def list_files(kind: Literal["train", "test"]):
    return state.get_files(kind, with_path=False)

@app.route("/episode/train/<episode_number>")
def get_train_episode(episode_num):
    file = state.get_episode_file("train", int(episode_num))
    return upload_file(file)

@app.route("/episode/test/<step_number>/<episode_number>")
def get_episode(episode_number: str, step_number):
    episode_number = int(episode_number)
    step_number = int(step_number)
    file = state.get_episode_file("test", int(episode_number), int(step_number))
    contents = upload_file(file)
    data = json.loads(contents)
    return data


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
    
        


def upload_file(filename: str) -> bytes:
    with open(filename, "rb") as f:
        return f.read()


def run(port=5174, debug=False):
    app.run("0.0.0.0", port=port, debug=debug)