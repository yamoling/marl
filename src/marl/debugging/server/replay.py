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
        self.experiment = Experiment.load(log_dir)

    def experiment_summary(self) -> tuple[list[ReplayEpisode], list[ReplayEpisode]]:
        return self.experiment.train_summary(), self.experiment.test_summary()
    
    def experiment_obs_type(self) -> str:
        return self.experiment.obs_type.name
    
    def env_info(self) -> dict:
        return self.experiment.env_info
    
    def get_tests_at(self, test_directory: str) -> list[ReplayEpisode]:
        return self.experiment.test_episode_summary(test_directory)

    def get_files(self, kind: Literal["train", "test"]) -> list[str]:
        if self.experiment is None:
            return []
        match kind:
            case "test": return self.experiment.list_tests()
            case "train":return self.experiment.list_trainings()
            case other: raise ValueError()

    def get_episode(self, directory: str) -> ReplayEpisode:
        return self.experiment.replay_episode(directory)
