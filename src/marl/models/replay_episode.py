import os
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
from marlenv import Episode, Space

from .action import Action

if TYPE_CHECKING:
    from marl.agents import ReplayAgent


@dataclass
class LightEpisodeSummary:
    rundir: str
    metrics: dict[str, float]
    test_num: int
    time_step: int

    def __init__(self, rundir: str, metrics: dict[str, float], time_step: int, test_num: int):
        self.rundir = rundir
        self.time_step = time_step
        self.metrics = metrics
        self.test_num = test_num
        self.name = os.path.basename(rundir)


@dataclass
class ReplayEpisode(LightEpisodeSummary):
    episode: Episode
    frames: list[str]
    agent_details: list[dict[str, Any]]
    action_space: Space
    replay_kind: str
    replay_mismatch: bool
    mismatch_details: list[str]

    def __init__(
        self,
        rundir: str,
        time_step: int,
        test_num: int,
        episode: Episode,
        frames: list[str],
        detailed_actions: list[Action],
        action_space: Space,
        replay_agent: "ReplayAgent",
    ):
        super().__init__(rundir, episode.metrics, time_step, test_num)
        self.episode = episode
        self.frames = frames
        self.agent_details = [a.details for a in detailed_actions]
        self.action_space = action_space
        self.replay_mismatch = replay_agent.mismatch
        self.mismatch_details = replay_agent.mismatch_details
        self.replay_kind = replay_agent.__class__.__name__
