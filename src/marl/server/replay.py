from marl.models import Experiment, ReplayEpisode


def get_episode(directory: str) -> ReplayEpisode:
    return Experiment.replay_episode(directory)
