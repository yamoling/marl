import pickle
from collections import defaultdict
from dataclasses import KW_ONLY, dataclass
from pathlib import Path

import numpy as np
from marlenv import Transition

from marl.models import Policy, Trainer
from marl.models.agent import Agent


@dataclass
class QLearning(Trainer):
    n_actions: int
    n_agents: int
    _: KW_ONLY
    lr: float = 0.1
    default_qvalue: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        self._qtable = defaultdict(self._init_qvalue)

    def _init_qvalue(self):  # -> npt.NDArray[np.float32]:
        return np.full((self.n_agents, self.n_actions), self.default_qvalue, dtype=np.float32)

    def update_step(self, transition: Transition, time_step: int):
        """Update chosen actions using legal successors and true terminal masking. @ai-generated"""
        actions = np.asarray(transition.action).reshape(self.n_agents)
        agents = np.arange(self.n_agents)
        qmatrix = self._qtable[transition.obs]
        target = np.full(self.n_agents, transition.reward.item(), dtype=np.float32)
        if not transition.done:
            next_qvalues = np.where(transition.next_obs.available_actions, self._qtable[transition.next_obs], -np.inf)
            target += self.gamma * next_qvalues.max(axis=-1)
        qmatrix[agents, actions] += self.lr * (target - qmatrix[agents, actions])
        return {}

    def save(self, directory: Path):
        import os

        qtable_file = os.path.join(directory, "qlearning.pkl")
        with open(qtable_file, "wb") as f:
            pickle.dump(self, f)

    def load(self, directory: Path):
        """Load current checkpoints, retaining the legacy filename fallback. @ai-generated"""
        import os

        file = os.path.join(directory, "qlearning.pkl")
        if not os.path.exists(file):
            file = os.path.join(directory, "qtable.pkl")
        with open(file, "rb") as f:
            loaded: QLearning = pickle.load(f)
        self._qtable = loaded._qtable
        self.n_actions = loaded.n_actions
        self.n_agents = loaded.n_agents
        self.gamma = loaded.gamma
        self.lr = loaded.lr
        self.default_qvalue = loaded.default_qvalue

    def make_agent(self, policy: Policy | None = None, test_policy: Policy | None = None) -> Agent:
        from marl.agents import QAgent
        from marl.policy import ArgMax, EpsilonGreedy

        if policy is None:
            policy = EpsilonGreedy.constant(0.1)
        if test_policy is None:
            test_policy = ArgMax()
        return QAgent(self._qtable, policy, test_policy)
