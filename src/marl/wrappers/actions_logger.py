import os
import json
from rlenv.models import Episode
from .deep_qwrapper import DeepQWrapper, IDeepQLearning


class ActionsLogger(DeepQWrapper):
    def __init__(self, wrapped: IDeepQLearning, logdir: str) -> None:
        super().__init__(wrapped)
        self.logdir = logdir
        self._training = True

    def test_directory(self, time_step: int, test_num: int = None) -> str:
        directory = os.path.join(self.logdir, "test", f"{time_step}")
        if test_num is not None:
            directory = os.path.join(directory, f"{test_num}")
        return directory
    
    def after_tests(self, episodes: list[Episode], time_step: int):
        for test_num, episode in enumerate(episodes):
            directory = self.test_directory(time_step, test_num)
            with open(os.path.join(directory, "actions.json"), "w") as f:
                json.dump(episode.actions.tolist(), f)
        return super().after_tests(episodes, time_step)
