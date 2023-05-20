import os
from dataclasses import dataclass
from rlenv.models import Metrics, Episode


@dataclass
class ReplayEpisodeSummary:
    directory: str
    metrics: Metrics

    @property
    def name(self) -> str:
        return os.path.basename(self.directory)

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "directory": self.directory,
            "metrics": self.metrics.to_json(),
        }

@dataclass
class ReplayEpisode(ReplayEpisodeSummary):
    episode: Episode
    qvalues: list[list[list[float]]]
    state_values: list[float]
    frames: list[str]

    @property
    def name(self) -> str:
        return os.path.basename(self.directory)

    def to_json(self) -> dict:
        return {
            **super().to_json(),
            "episode": self.episode.to_json(),
            "qvalues": self.qvalues,
            "frames": self.frames,
            "state_values": self.state_values
        }