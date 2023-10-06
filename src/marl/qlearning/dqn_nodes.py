import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from rlenv.models import Observation, Transition

from marl.models import RLAlgo
from marl.policy import EpsilonGreedy, Policy
from marl.training import DQNTrainer
from marl.utils import defaults_to


@dataclass
class DQN(RLAlgo):
    """
    Independent Deep Q-Network agent with shared QNetwork.
    If agents require different behaviours, an agentID should be included in the
    observation 'extras'.
    """

    trainer: DQNTrainer
    train_policy: Optional[Policy] = None
    test_policy: Optional[Policy] = None

    def __post_init__(self):
        self.qnetwork = self.trainer.qnetwork
        self.device = self.qnetwork.device
        self.train_policy = defaults_to(self.train_policy, lambda: EpsilonGreedy.constant(0.1))
        self.test_policy = defaults_to(self.test_policy, lambda: EpsilonGreedy.constant(0.05))
        self.policy = self.train_policy
        self.device = self.qnetwork.device
        self._train_logs = {}

    @torch.no_grad()
    def choose_action(self, obs: Observation) -> np.ndarray:
        qvalues = self.compute_qvalues(obs)
        qvalues = qvalues.cpu().numpy()
        return self.policy.get_action(qvalues, obs.available_actions)

    @torch.no_grad()
    def value(self, obs: Observation) -> float:
        qvalues = self.compute_qvalues(obs)
        max_qvalues = torch.max(qvalues, dim=-1).values
        return torch.mean(max_qvalues).item()

    def after_train_step(self, transition: Transition, time_step: int):
        self.trainer.update_step(transition, time_step)

    def compute_qvalues(self, obs: Observation) -> torch.Tensor:
        obs_data = torch.from_numpy(obs.data).to(self.device, non_blocking=True)
        obs_extras = torch.from_numpy(obs.extras).to(self.device, non_blocking=True)
        return self.qnetwork.forward(obs_data, obs_extras)

    def before_tests(self, time_step: int):
        self.policy = self.test_policy

    def after_tests(self, time_step: int, episodes):
        self.policy = self.train_policy

    def save(self, to_directory: str):
        os.makedirs(to_directory, exist_ok=True)
        qnetwork_path = os.path.join(to_directory, "qnetwork.weights")
        train_policy_path = os.path.join(to_directory, "train_policy")
        test_policy_path = os.path.join(to_directory, "test_policy")
        torch.save(self.qnetwork.state_dict(), qnetwork_path)
        self.train_policy.save(train_policy_path)
        self.test_policy.save(test_policy_path)

    def load(self, from_directory: str):
        qnetwork_path = os.path.join(from_directory, "qnetwork.weights")
        train_policy_path = os.path.join(from_directory, "train_policy")
        test_policy_path = os.path.join(from_directory, "test_policy")
        self.qnetwork.load_state_dict(torch.load(qnetwork_path))
        self.train_policy = self.train_policy.__class__.load(train_policy_path)
        self.test_policy = self.test_policy.__class__.load(test_policy_path)
        self.policy = self.train_policy

    def to(self, device: torch.device):
        self.qnetwork.to(device)
        self.trainer.to(device)
        self.device = device

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        import marl

        data["trainer"] = marl.training.from_dict(data["trainer"])
        data["train_policy"] = marl.policy.from_dict(data["train_policy"])
        data["test_policy"] = marl.policy.from_dict(data["test_policy"])
        return super().from_dict(data)
