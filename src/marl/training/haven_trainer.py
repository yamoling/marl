from copy import copy
from typing import Literal

import numpy as np
from marlenv import Episode, Transition

from marl.agents import Haven
from marl.models.trainer import Trainer


class HavenTrainer(Trainer):
    def __init__(
        self,
        meta_trainer: Trainer,
        worker_trainer: Trainer,
        k: int,
        n_subgoals: int,
    ):
        super().__init__("step")
        self.meta_trainer = meta_trainer
        self.worker_trainer = worker_trainer
        self.k = k
        self.cumulative_reward = np.zeros(0, dtype=np.float32)
        self.n_subgoals = n_subgoals

    def update_step(self, transition: Transition, time_step: int):
        logs = dict[str, float]()
        worker_logs = self.worker_trainer.update_step(transition, time_step)
        for key, value in worker_logs.items():
            logs[f"worker-{key}"] = value
        if time_step == 0:
            self.cumulative_reward = transition.reward.copy()
        elif time_step % self.k == 0 or transition.is_terminal:
            meta_transition = copy(transition)
            meta_transition.action = transition["meta_actions"]
            meta_transition.reward = self.cumulative_reward
            meta_logs = self.meta_trainer.update_step(transition, time_step)
            self.cumulative_reward = np.zeros_like(transition.reward)
            for key, value in meta_logs.items():
                logs[f"meta-{key}"] = value
        else:
            self.cumulative_reward += transition.reward
        return logs

    def update_episode(self, episode: Episode, episode_num: int, time_step: int):
        return self.worker_trainer.update_episode(episode, episode_num, time_step)

    def make_agent(self):
        return Haven(
            self.meta_trainer.make_agent(),
            self.worker_trainer.make_agent(),
            self.n_subgoals,
            self.k,
        )

    def randomize(self, method: Literal["xavier", "orthogonal"] = "xavier"):
        self.meta_trainer.randomize(method)
        self.worker_trainer.randomize(method)
