from dataclasses import dataclass
from copy import copy
from typing import Literal

import numpy as np
from marlenv import Observation
from torch import device

from ..agent import Agent


@dataclass
class Haven(Agent[tuple[np.ndarray, dict[str, np.ndarray]]]):
    """
    Hierarchical agent where a meta-agent gives orders (i.e. actions, subgoals, communication) to workers (multi-agent).

    The subgoals are concatenated to the worker's observations as "extras".
    """

    meta: Agent[np.ndarray]
    workers: Agent[np.ndarray]
    n_subgoals: int
    k: int
    """The number of steps that meta-actions lasts."""

    def __init__(
        self,
        meta_agent: Agent,
        workers: Agent[np.ndarray],
        n_subgoals: int,
        n_workers: int,
        k: int,
        n_meta_extras: int,
        n_agent_extras: int,
    ):
        super().__init__()
        self.meta = meta_agent
        self.workers = workers
        self.k = k
        self.n_subgoals = n_subgoals
        self._t_train = 0
        self._t_test = 0
        self._t = 0
        self.last_meta_action = np.zeros(0, dtype=np.float32)
        self.last_subgoals = np.zeros(0, dtype=np.float32)
        self.n_meta_extras = n_meta_extras
        self.n_agent_extras = n_agent_extras
        self._all_meta_actions_avaiable = np.full((n_workers, self.n_subgoals), True)
        self.n_workers = n_workers

    def choose_action(self, observation: Observation):
        assert observation.extras_shape[0] == self.n_meta_extras + self.n_agent_extras + self.n_subgoals
        if self._t % self.k == 0:
            meta_obs = self.make_meta_observation(observation)
            meta_action = self.meta.choose_action(meta_obs)
            if meta_action.ndim == 1:
                # Encode discrete actions as one-hot
                subgoals = np.eye(self.n_subgoals, dtype=np.float32)[meta_action]
            else:
                subgoals = meta_action
            self.last_meta_action = meta_action
            self.last_subgoals = subgoals
        self._t += 1
        observation.extras[:, -self.n_subgoals :] = self.last_subgoals
        workers_actions = self.workers.choose_action(observation)
        return workers_actions, {"meta_actions": self.last_meta_action}

    def make_meta_observation(self, observation: Observation):
        meta_obs = copy(observation)
        # remove the subgoals padding
        meta_obs.extras = observation.extras[:, : self.n_meta_extras]
        # All actions (i.e. subgoals) are always available
        meta_obs.available_actions = self._all_meta_actions_avaiable
        return meta_obs

    def new_episode(self):
        super().new_episode()
        self.workers.new_episode()
        self.meta.new_episode()
        self._t = 0

    def set_testing(self):
        self.meta.set_testing()
        self.workers.set_testing()
        # Save the train time step and restore the test time step
        self._t_train = self._t
        self._t = self._t_test

    def set_training(self):
        self.meta.set_training()
        self.workers.set_training()
        # Save the test time step and restore the train time step
        self._t_test = self._t
        self._t = self._t_train

    def to(self, device: device):
        self.device = device
        self.meta.to(device)
        self.workers.to(device)
        return self

    def randomize(self, method: Literal["xavier", "orthogonal"] = "xavier"):
        self.meta.randomize(method)
        self.workers.randomize(method)
