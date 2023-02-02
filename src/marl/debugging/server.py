from dataclasses import dataclass
import os
import json
import base64
import cv2
from typing import Literal
from flask import Flask
from flask_cors import CORS
from marl.utils import alpha_num_order

app = Flask(__name__)
CORS(app)

@dataclass
class ServerState:
    replay_dir: str = None

    @property
    def train_dir(self) -> str:
        return os.path.join(self.replay_dir, "train/")

    @property
    def test_dir(self) -> str:
        return os.path.join(self.replay_dir, "test/")

    def get_files(self, kind: Literal["train", "test"], with_path=False) -> list[str]:
        match kind:
            case "train": 
                folder = self.train_dir
            case "test": 
                folder = self.test_dir
            case _: raise ValueError()
        files = sorted(os.listdir(folder), key=alpha_num_order)
        if kind == "test":
            files = [f for f in files if f.startswith("step")]
        if with_path:
            files = [os.path.join(folder, file) for file in files]
        return files


state = ServerState("files/")


@app.route("/list/<kind>")
def list_files(kind: Literal["train", "test"]):
    return state.get_files(kind)

@app.route("/episode/<kind>/<number>")
def get_episode(kind: str, number: str):
    match kind:
        case "train": 
            file = os.path.join(state.train_dir,  f"episode-{number}.json")
        case "test":
            step_dir = list_files("test")
            step_dir = step_dir[int(number)]
            file = os.path.join(state.test_dir, step_dir, f"0.json")
        case other: raise ValueError(f"Unknown kind {other}")
    return upload_file(file)


@app.route("/metrics/<kind>")
def get_metrics(kind: str):
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
            
@app.route("/frames/<kind>/<episode_num>")
def get_frames(kind: Literal["test", "train"], episode_num: str):
    episode_num = int(episode_num)
    cap = cv2.VideoCapture("path_to_file")
    _, frames = cap.read()
    frames = [base64.b64encode(cv2.imencode(".jpg", f)[1]).decode("ascii") for f in frames]
    return frames
    
        


def upload_file(filename: str) -> bytes:
    with open(filename, "rb") as f:
        return f.read()


def run(port=5174):
    app.run("0.0.0.0", port=port, debug=True)