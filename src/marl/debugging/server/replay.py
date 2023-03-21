from dataclasses import dataclass
from marl.models import Experiment, ReplayEpisode


@dataclass
class ReplayServerState:
    experiment: Experiment | None

    def __init__(self) -> None:
        self.experiment = None

    def update(self, log_dir: str):
        self.experiment = Experiment.load(log_dir)

    def experiment_summary(self) -> tuple[list[ReplayEpisode], list[ReplayEpisode]]:
        return self.experiment.train_summary(), self.experiment.test_summary()
    
    def env_info(self) -> dict:
        return self.experiment.env_info
    
    def get_tests_at(self, test_directory: str) -> list[ReplayEpisode]:
        return self.experiment.test_episode_summary(test_directory)

    def get_episode(self, directory: str) -> ReplayEpisode:
        return self.experiment.replay_episode(directory)
