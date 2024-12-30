from copy import copy
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from marlenv import Episode, Transition
from torch import device

from marl.agents import Haven
from marl.models.nn import CriticNN, Mixer
from marl.models.trainer import Trainer
from marl.models import TransitionMemory


@dataclass
class HavenTrainer(Trainer):
    def __init__(
        self,
        meta_trainer: Trainer,
        value_network: CriticNN,
        value_mixer: Mixer,
        worker_trainer: Trainer,
        n_workers: int,
        n_subgoals: int,
        k: int,
        n_meta_extras: int,
        n_agent_extras: int,
        n_meta_warmup_steps: int,
        gamma: float,
        memory_size: int = 50_000,
        batch_size: int = 32,
        value_lr: float = 1e-4,
    ):
        super().__init__("step")
        self.meta_trainer = meta_trainer
        self.worker_trainer = worker_trainer
        self.value_network = value_network
        self.k = k
        self.cumulative_reward = np.zeros(0, dtype=np.float32)
        self.n_subgoals = n_subgoals
        self.n_meta_extras = n_meta_extras
        self.n_agent_extras = n_agent_extras
        self.n_workers = n_workers
        self.available_actions = np.full((self.n_workers, self.n_subgoals), True)
        self.n_warmup_steps = n_meta_warmup_steps
        self.batch_size = batch_size
        self.value_memory = TransitionMemory(memory_size)
        self.gamma = gamma
        self.mixer = value_mixer
        self.value_optimizer = torch.optim.Adam(list(self.value_network.parameters()) + list(value_mixer.parameters()), lr=value_lr)

    def update_step(self, transition: Transition, time_step: int):
        logs = dict[str, float]()
        for key, value in self.worker_trainer.update_step(transition, time_step).items():
            logs[f"worker-{key}"] = value
        meta_transition = self._build_meta_transition(transition, time_step)
        if meta_transition is not None:
            logs |= self._train_value_network(meta_transition)
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

    def _train_value_network(self, meta_transition: Transition):
        self.value_memory.add(meta_transition)
        if not self.value_memory.can_sample(self.batch_size):
            return {}
        batch = self.value_memory.sample(self.batch_size).to(self.device)
        values = self.value_network.value(batch.states, batch.states_extras)
        with torch.no_grad():
            next_values = self.meta_trainer.next_values(batch)
        targets = batch.rewards + self.gamma * next_values
        loss = torch.nn.functional.mse_loss(values, targets)
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()
        return {"value_loss": float(loss.item())}

    def update_episode(self, episode: Episode, episode_num: int, time_step: int):
        return self.worker_trainer.update_episode(episode, episode_num, time_step)

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
        self.value_network.randomize(method)

    def to(self, device: device):
        self.meta_trainer = self.meta_trainer.to(device)
        self.worker_trainer = self.worker_trainer.to(device)
        self.value_network = self.value_network.to(device)
        self.device = device
        return self
