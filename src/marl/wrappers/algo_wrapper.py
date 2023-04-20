import numpy as np
from rlenv import Observation, Transition, Episode
from marl.models import RLAlgo


class AlgoWrapper(RLAlgo):
    def __init__(self, wrapped: RLAlgo) -> None:
        super().__init__()
        self.algo = wrapped

    def choose_action(self, observation: Observation) -> np.ndarray[np.int64]:
        return self.algo.choose_action(observation)

    def save(self, to_path: str):
        return self.algo.save(to_path)

    def load(self, from_path: str):
        return self.algo.load(from_path)

    def before_train_episode(self, episode_num: int):
        return self.algo.before_train_episode(episode_num)
    
    def before_tests(self, time_step: int):
        return self.algo.before_tests(time_step)

    def after_train_episode(self, episode_num: int, episode: Episode):
        return self.algo.after_train_episode(episode_num, episode)

    def after_train_step(self, transition: Transition, time_step: int):
        return self.algo.after_train_step(transition, time_step)

    def after_tests(self, episodes: list[Episode], time_step: int):
        return self.algo.after_tests(episodes, time_step)
    
    def before_test_episode(self, time_step: int, test_num: int):
        return self.algo.before_test_episode(time_step, test_num)
    
    def after_test_episode(self, time_step: int, test_num: int, episode: Episode):
        return self.algo.after_test_episode(time_step, test_num, episode)
    
    def summary(self) -> dict:
        return self.algo.summary()