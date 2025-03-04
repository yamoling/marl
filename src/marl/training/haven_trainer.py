from copy import copy
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from marlenv import Episode, Transition
from torch import device

from marl.agents import Haven
from marl.models.trainer import Trainer


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
    ):
        super().__init__("step")
        self.meta_trainer = meta_trainer
        self.worker_trainer = worker_trainer
        self.k = k
        self.n_subgoals = n_subgoals
        self.n_meta_extras = n_meta_extras
        self.n_agent_extras = n_agent_extras
        self.n_workers = n_workers
        self.n_warmup_steps = n_meta_warmup_steps
        self._available_actions = np.full((self.n_workers, self.n_subgoals), True)
        self._cumulative_reward = np.zeros(0, dtype=np.float32)
        self._episode_step = 0

    def update_step(self, transition: Transition, time_step: int):
        logs = dict[str, float]()
        for key, value in self.worker_trainer.update_step(transition, time_step).items():
            logs[f"worker-{key}"] = value
        if time_step < self.n_warmup_steps:
            self._cumulative_reward = np.zeros_like(transition.reward)
            return logs
        if self.meta_trainer.update_on_steps:
            meta_transition = self._build_meta_transition(transition)
            if meta_transition is not None:
                meta_logs = self.meta_trainer.update_step(meta_transition, time_step)
                for key, value in meta_logs.items():
                    logs[f"meta-{key}"] = value
        return logs

    def update_episode(self, episode: Episode, episode_num: int, time_step: int):
        logs = dict[str, float]()
        for key, value in self.worker_trainer.update_episode(episode, episode_num, time_step).items():
            logs[f"worker-{key}"] = value
        if self.meta_trainer.update_on_episodes and time_step >= self.n_warmup_steps:
            meta_episode = self._build_meta_episode(episode)
            meta_logs = self.meta_trainer.update_episode(meta_episode, episode_num, time_step)
            for key, value in meta_logs.items():
                logs[f"meta-{key}"] = value
        return logs

    def _build_meta_transition(self, transition: Transition) -> Transition | None:
        # A meta-transition is built every k steps of an episode.
        # Every k steps, we reset the cumulative reward to 0.
        # If the episode terminates, we reset the episode step to 0.
        if self._episode_step % self.k == 0 or transition.is_terminal:
            meta_transition = self.make_meta_transition(transition)
            self._cumulative_reward.fill(0.0)
        else:
            meta_transition = None
            self._cumulative_reward += transition.reward
        if transition.is_terminal:
            self._episode_step = 0
        else:
            self._episode_step += 1
        return meta_transition

    def _build_meta_episode(self, episode: Episode) -> Episode:
        raise Exception("Not tested yet !")
        transitions = list(episode.transitions())
        meta_episode = Episode.new(transitions[0].obs, transitions[0].state)
        reward = np.zeros_like(transitions[0].reward)
        for t, worker_transition in enumerate(transitions[1:], 1):
            if t % self.k == 0:
                meta_transition = self.make_meta_transition(worker_transition, reward)
                meta_episode.add(meta_transition)
                reward = np.zeros_like(worker_transition.reward)
            else:
                reward += worker_transition.reward
        return meta_episode

    def make_meta_transition(self, worker_transition: Transition, cumulative_reward: Optional[np.ndarray] = None) -> Transition:
        # Avoid overhead with shallow copy
        obs = copy(worker_transition.obs)
        next_obs = copy(worker_transition.next_obs)

        # Only keep the meta-extras
        obs.extras = obs.extras[:, : self.n_meta_extras]
        next_obs.extras = next_obs.extras[:, : self.n_meta_extras]
        # All actions are always available
        obs.available_actions = self._available_actions
        next_obs.available_actions = self._available_actions
        if cumulative_reward is None:
            cumulative_reward = self._cumulative_reward

        return Transition(
            obs=obs,
            next_obs=next_obs,
            state=worker_transition.state,
            next_state=worker_transition.next_state,
            action=worker_transition["meta_actions"],  # The action is the meta agent's action
            reward=cumulative_reward,  # The reward is the cumulative reward over the last k steps
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
