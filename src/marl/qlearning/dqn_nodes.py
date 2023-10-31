import os
import pickle
from dataclasses import dataclass
from typing import Optional
from serde import serde

import numpy as np
import torch
from rlenv.models import Observation

from marl.nn import LinearNN
from marl.models import RLAlgo
from marl.policy import Policy

@serde
@dataclass
class DQN(RLAlgo):
    """
    Deep Q-Network agent with shared QNetwork.
    """

    qnetwork: LinearNN
    train_policy: Policy
    test_policy: Policy

    def __init__(self, qnetwork: LinearNN, train_policy: Policy, test_policy: Optional[Policy] = None):
        super().__init__()
        self.qnetwork = qnetwork
        self.device = self.qnetwork.device
        self.train_policy = train_policy
        if test_policy is None:
            test_policy = self.train_policy
        self.test_policy = test_policy
        self.policy = self.train_policy

    def choose_action(self, obs: Observation) -> np.ndarray:
        with torch.no_grad():
            qvalues = self.compute_qvalues(obs)
        qvalues = qvalues.cpu().numpy()
        return self.policy.get_action(qvalues, obs.available_actions)

    def value(self, obs: Observation) -> float:
        qvalues = self.compute_qvalues(obs)
        max_qvalues = torch.max(qvalues, dim=-1).values
        return torch.mean(max_qvalues).item()

    def compute_qvalues(self, obs: Observation) -> torch.Tensor:
        obs_data = torch.from_numpy(obs.data).to(self.device, non_blocking=True)
        obs_extras = torch.from_numpy(obs.extras).to(self.device, non_blocking=True)
        return self.qnetwork.forward(obs_data, obs_extras)

    def set_testing(self):
        self.policy = self.test_policy

    def set_training(self):
        self.policy = self.train_policy

    def save(self, to_directory: str):
        os.makedirs(to_directory, exist_ok=True)
        torch.save(self.qnetwork.state_dict(), f"{to_directory}/qnetwork.weights")
        train_policy_path = os.path.join(to_directory, "train_policy")
        test_policy_path = os.path.join(to_directory, "test_policy")
        with (open(train_policy_path, "wb") as f,
              open(test_policy_path, "wb") as g):
            pickle.dump(self.train_policy, f)
            pickle.dump(self.test_policy, g)

    def load(self, from_directory: str):
        self.qnetwork.load_state_dict(torch.load(f"{from_directory}/qnetwork.weights"))
        train_policy_path = os.path.join(from_directory, "train_policy")
        test_policy_path = os.path.join(from_directory, "test_policy")
        with (open(train_policy_path, "rb") as f,
              open(test_policy_path, "rb") as g):
            self.train_policy = pickle.load(f)
            self.test_policy = pickle.load(g)
        self.policy = self.train_policy

    def randomize(self):
        self.qnetwork.randomize()

    def to(self, device: torch.device):
        self.qnetwork.to(device)
        self.device = device
