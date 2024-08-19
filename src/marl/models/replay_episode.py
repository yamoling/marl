import os
from dataclasses import dataclass
from rlenv.models import Episode
from serde import serde
from typing import Optional


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
    qvalues: Optional[list[list[list[float]]]]
    state_values: list[float]
    frames: list[str]
    logits: Optional[list[list[list[float]]]]
    probs: Optional[list[list[list[float]]]]
    messages: Optional[list[list[list[float]]]]
    received_messages: Optional[list[list[float]]]
    init_qvalues: Optional[list[list[list[float]]]]

    def __init__(
        self,
        directory: str,
        metrics: dict[str, float],
        episode: Episode,
        state_values: list[float],
        frames: list[str],
        qvalues: Optional[list[list[list[float]]]] = None,
        logits: Optional[list[list[list[float]]]] = None,
        probs: Optional[list[list[list[float]]]] = None,
        messages: Optional[list[list[list[float]]]] = None,
        received_messages: Optional[list[list[float]]] = None,
        init_qvalues: Optional[list[list[list[float]]]] = None,
    ):
        super().__init__(directory, metrics)
        self.episode = episode
        self.qvalues = qvalues
        self.state_values = state_values
        self.frames = frames
        self.logits = logits
        self.probs = probs
        self.messages = messages
        self.received_messages = received_messages
        self.init_qvalues = init_qvalues
