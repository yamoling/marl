from dataclasses import dataclass
from typing import Literal
from rlenv.models import Metrics
from marl.models import Experiment, ReplayEpisode


@dataclass
class Item:
    filename: str
    metrics: Metrics

@dataclass
class ReplayServerState:
    experiment: Experiment | None

    def __init__(self, replay_dir: str=None) -> None:
        self.experiment = None
        if replay_dir is not None:
            self.update(replay_dir)

    def update(self, log_dir: str):
        self.experiment = Experiment(log_dir)

    def experiment_summary(self) -> tuple[list[ReplayEpisode], list[ReplayEpisode]]:
        return self.experiment.train_summary(), self.experiment.test_summary()

    def get_files(self, kind: Literal["train", "test"]) -> list[str]:
        if self.experiment is None:
            return []
        match kind:
            case "test": return self.experiment.list_tests()
            case "train":return self.experiment.list_trainings()
            case other: raise ValueError()

    def get_episode(self, directory: str) -> ReplayEpisode:
        return self.experiment.replay_episode(directory)

    def get_train_episode(self, episode_num: int) -> ReplayEpisode:
        return self.experiment.replay_train_episode(episode_num)

    def get_test_episodes(self, step_num: int) -> list[str]:
        return self.experiment.list_test_episodes(step_num)
    

