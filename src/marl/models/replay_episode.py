import os
from dataclasses import dataclass, field
from marlenv.models import Episode
from typing import Literal

from .detailed_action import DetailedAction


@dataclass
class LightEpisodeSummary:
    name: str = field(init=False)
    directory: str
    metrics: dict[str, float]

    def __init__(self, directory: str, metrics: dict[str, float]):
        self.directory = directory
        self.metrics = metrics
        self.name = os.path.basename(directory)


@dataclass
class DecisionData:
    label: str
    data: list[list[list[float]]]


@dataclass
class ReplayEpisode(LightEpisodeSummary):
    episode: Episode
    frames: list[str]
    decision_data: DecisionData

    def __init__(self, directory: str, episode: Episode, frames: list[str], detailed_actions: list[DetailedAction]):
        super().__init__(directory=directory, metrics=episode.metrics)
        self.episode = episode
        self.frames = frames
        label = detailed_actions[0].label
        self.decision_data = DecisionData(label, data=[action.details.tolist() for action in detailed_actions])
