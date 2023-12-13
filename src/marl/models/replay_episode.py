import os
from dataclasses import dataclass
from rlenv.models import Episode


@dataclass
class ReplayEpisodeSummary:
    directory: str
    metrics: dict

    @property
    def name(self) -> str:
        return os.path.basename(self.directory)

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "directory": self.directory,
            "metrics": self.metrics,
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
            "state_values": self.state_values,
        }
