import os
import pickle
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
    """

    trainer: DQNTrainer
    train_policy: Optional[Policy] = None
    test_policy: Optional[Policy] = None

    def __init__(self, trainer: DQNTrainer, train_policy: Optional[Policy] = None, test_policy: Optional[Policy] = None):
        super().__init__()
        self.trainer = trainer
        self.qnetwork = self.trainer.qnetwork
        self.device = self.qnetwork.device
        self.train_policy = defaults_to(train_policy, lambda: EpsilonGreedy.constant(0.1))
        self.test_policy = defaults_to(test_policy, lambda: EpsilonGreedy.constant(0.01))
        self.policy = self.train_policy
        self.device = self.qnetwork.device

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
        self.trainer.save(to_directory)
        train_policy_path = os.path.join(to_directory, "train_policy")
        test_policy_path = os.path.join(to_directory, "test_policy")
        with (open(train_policy_path, "wb") as f,
              open(test_policy_path, "wb") as g):
            pickle.dump(self.train_policy, f)
            pickle.dump(self.test_policy, g)

    def load(self, from_directory: str):
        self.trainer.load(from_directory)
        self.qnetwork = self.trainer.qnetwork
        train_policy_path = os.path.join(from_directory, "train_policy")
        test_policy_path = os.path.join(from_directory, "test_policy")
        with (open(train_policy_path, "rb") as f,
              open(test_policy_path, "rb") as g):
            self.train_policy = pickle.load(f)
            self.test_policy = pickle.load(g)
        self.policy = self.train_policy


    def to(self, device: torch.device):
        self.qnetwork.to(device)
        self.trainer.to(device)
        self.device = device
