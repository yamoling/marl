import os
from dataclasses import dataclass
from rlenv.models import Episode
from serde import serde


@serde
@dataclass
class ReplayEpisodeSummary:
    name: str
    directory: str
    metrics: dict[str, float]

    def __init__(self, directory: str, metrics: dict[str, float]):
        self.directory = directory
        self.metrics = metrics
        self.name = os.path.basename(directory)


@serde
@dataclass
class ReplayEpisode(ReplayEpisodeSummary):
    episode: Episode
    qvalues: list[list[list[float]]]
    state_values: list[float]
    frames: list[str]

    def __init__(
        self,
        directory: str,
        metrics: dict[str, float],
        episode: Episode,
        qvalues: list[list[list[float]]],
        state_values: list[float],
        frames: list[str],
    ):
        super().__init__(directory, metrics)
        self.episode = episode
        self.qvalues = qvalues
        self.state_values = state_values
        self.frames = frames
