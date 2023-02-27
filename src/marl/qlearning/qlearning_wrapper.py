import torch
import numpy as np
from rlenv import Observation, Episode, Transition
from marl.models import Batch
from .qlearning import IDeepQLearning


class DeepQWrapper(IDeepQLearning):
    def __init__(self, wrapped: IDeepQLearning) -> None:
        IDeepQLearning.__init__(self)
        self.algo = wrapped

    @property
    def gamma(self) -> float:
        return self.algo.gamma

    def compute_qvalues(self, data):
        return self.algo.compute_qvalues(data)

    def compute_loss(self, qvalues: torch.Tensor, qtargets: torch.Tensor, batch: Batch) -> torch.Tensor:
        return self.algo.compute_loss(qvalues, qtargets, batch)

    def compute_targets(self, batch: Batch) -> torch.Tensor:
        return self.algo.compute_targets(batch)

    def process_batch(self, batch: Batch) -> Batch:
        return self.algo.process_batch(batch)
    
    def choose_action(self, observation: Observation) -> np.ndarray[np.int64]:
        return self.algo.choose_action(observation)

    def save(self, to_path: str):
        return self.algo.save(to_path)

    def load(self, from_path: str):
        return self.algo.load(from_path)

    def before_episode(self, episode_num: int):
        return self.algo.before_episode(episode_num)
    
    def before_tests(self):
        return self.algo.before_tests()

    def after_episode(self, episode_num: int, episode: Episode):
        return self.algo.after_episode(episode_num, episode)

    def after_step(self, transition: Transition, step_num: int):
        return self.algo.after_step(transition, step_num)

    def after_tests(self, episodes: list[Episode], time_step: int):
        return self.algo.after_tests(episodes, time_step)
