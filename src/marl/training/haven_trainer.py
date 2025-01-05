from copy import copy
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from marlenv import Transition
from torch import device

from marl.agents import Haven
from marl.models.trainer import Trainer
from marl.models import TransitionMemory


@dataclass
class HavenTrainer(Trainer):
    def __init__(
        self,
        meta_trainer: Trainer,
        worker_trainer: Trainer,
        n_workers: int,
        n_subgoals: int,
        k: int,
        n_meta_extras: int,
        n_agent_extras: int,
        n_meta_warmup_steps: int,
        gamma: float,
        batch_size: int = 32,
    ):
        super().__init__("step")
        self.meta_trainer = meta_trainer
        self.worker_trainer = worker_trainer
        self.k = k
        self.cumulative_reward = np.zeros(0, dtype=np.float32)
        self.n_subgoals = n_subgoals
        self.n_meta_extras = n_meta_extras
        self.n_agent_extras = n_agent_extras
        self.n_workers = n_workers
        self.available_actions = np.full((self.n_workers, self.n_subgoals), True)
        self.n_warmup_steps = n_meta_warmup_steps
        self.batch_size = batch_size
        self.gamma = gamma

    def update_step(self, transition: Transition, time_step: int):
        logs = dict[str, float]()
        for key, value in self.worker_trainer.update_step(transition, time_step).items():
            logs[f"worker-{key}"] = value
        meta_transition = self._build_meta_transition(transition, time_step)
        if meta_transition is not None:
            if time_step >= self.n_warmup_steps:
                meta_logs = self.meta_trainer.update_step(meta_transition, time_step)
                for key, value in meta_logs.items():
                    logs[f"meta-{key}"] = value
        return logs

    def _build_meta_transition(self, transition: Transition, time_step: int) -> Transition | None:
        if time_step == 0:
            self.cumulative_reward = transition.reward.copy()
        elif time_step % self.k == 0 or transition.is_terminal:
            meta_transition = self.make_meta_transition(transition)
            self.cumulative_reward = np.zeros_like(transition.reward)
            return meta_transition
        else:
            self.cumulative_reward += transition.reward

    def make_meta_transition(self, worker_transition: Transition) -> Transition:
        # Avoid overhead with shallow copy
        obs = copy(worker_transition.obs)
        next_obs = copy(worker_transition.next_obs)

        # Only keep the meta-extras
        obs.extras = obs.extras[:, : self.n_meta_extras]
        next_obs.extras = next_obs.extras[:, : self.n_meta_extras]
        # All actions are always available
        obs.available_actions = self.available_actions
        next_obs.available_actions = self.available_actions

        return Transition(
            obs=obs,
            next_obs=next_obs,
            state=worker_transition.state,
            next_state=worker_transition.next_state,
            action=worker_transition["meta_actions"],  # The action is the meta agent's action
            reward=self.cumulative_reward,  # The reward is the cumulative reward over the last k steps
            done=worker_transition.done,
            truncated=worker_transition.truncated,
            info=worker_transition.info,
        )

    def make_agent(self):
        return Haven(
            meta_agent=self.meta_trainer.make_agent(),
            workers=self.worker_trainer.make_agent(),
            n_subgoals=self.n_subgoals,
            n_workers=self.n_workers,
            k=self.k,
            n_meta_extras=self.n_meta_extras,
            n_agent_extras=self.n_agent_extras,
        )

    def randomize(self, method: Literal["xavier", "orthogonal"] = "xavier"):
        self.meta_trainer.randomize(method)
        self.worker_trainer.randomize(method)

    def to(self, device: device):
        self.meta_trainer = self.meta_trainer.to(device)
        self.worker_trainer = self.worker_trainer.to(device)
        self.device = device
        return self
