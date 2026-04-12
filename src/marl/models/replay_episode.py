import os
from dataclasses import dataclass, field
from typing import Any
from marlenv import Episode, Space

from .action import Action


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
class ReplayEpisode(LightEpisodeSummary):
    episode: Episode
    frames: list[str]
    action_details: list[dict[str, Any]]
    action_space: Space

    def __init__(
        self,
        directory: str,
        episode: Episode,
        frames: list[str],
        detailed_actions: list[Action],
        action_space: Space,
    ):
        super().__init__(directory=directory, metrics=episode.metrics)
        self.episode = episode
        self.frames = frames
        self.action_details = [a.details for a in detailed_actions]
        self.action_space = action_space
