from dataclasses import dataclass
import os
import base64
import cv2
from typing import Literal
from rlenv.models import Metrics
from marl.utils import alpha_num_order
import marl


@dataclass
class TestItem:
    filename: str
    episodes: list[str]
    metrics: list[Metrics]

@dataclass
class ReplayServerState:
    train_episodes_paths: list[str]
    tests_folders_paths: list[str]
    runner: marl.debugging.DebugRunner
    _replay_dir: str = None

    def __init__(self, replay_dir: str=None) -> None:
        self._replay_dir = replay_dir
        self.train_episodes_paths = []
        self.tests_folders_paths = []
        self.algo = None
        self.runner = None
        if replay_dir is not None:
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
            case "test": files = self.tests_folders_paths
            case "train":files = self.train_episodes_paths
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

