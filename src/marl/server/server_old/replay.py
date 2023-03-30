from dataclasses import dataclass
from marl.models import Experiment, ReplayEpisode, ReplayEpisodeSummary
from dataclasses import dataclass


@dataclass
class ReplayServerState:
    experiment: Experiment | None

    def __init__(self) -> None:
        self.experiment = None

    def update(self, log_dir: str):
        self.experiment = Experiment.load(log_dir)

    def experiment_summary(self) -> tuple[list[ReplayEpisodeSummary], list[ReplayEpisodeSummary]]:
        return self.experiment.train_summary(), self.experiment.test_summary()
    
    def env_info(self) -> dict:
        return self.experiment.env_info
    
    @staticmethod
    def get_tests_at(self, test_directory: str) -> list[ReplayEpisodeSummary]:
        return self.experiment.get_test_episodes(test_directory)

    def get_episode(self, directory: str) -> ReplayEpisode:
        return self.experiment.replay_episode(directory)


def get_tests_at(test_directory: str) -> list[ReplayEpisodeSummary]:
    return Experiment.get_test_episodes(test_directory)

def get_episode(directory: str) -> ReplayEpisode:
    return Experiment.replay_episode(directory)
