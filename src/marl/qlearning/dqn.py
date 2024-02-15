import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Generic, TypeVar
from serde import serde

import numpy as np
import torch
from rlenv.models import Observation

from marl.models import RLAlgo, LinearNN, RecurrentNN, NN, Policy


N = TypeVar("N", bound=NN)


@serde
@dataclass
class IDQN(RLAlgo, ABC, Generic[N]):
    """
    Deep Q-Network Interface with shared QNetwork.
    """

    qnetwork: N
    train_policy: Policy
    test_policy: Policy

    def __init__(self, qnetwork: N, train_policy: Policy, test_policy: Optional[Policy] = None):
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

    @abstractmethod
    def compute_qvalues(self, obs: Observation) -> torch.Tensor:
        pass

    def set_testing(self):
        self.policy = self.test_policy

    def set_training(self):
        self.policy = self.train_policy

    def save(self, to_directory: str):
        os.makedirs(to_directory, exist_ok=True)
        torch.save(self.qnetwork.state_dict(), f"{to_directory}/qnetwork.weights")
        train_policy_path = os.path.join(to_directory, "train_policy")
        test_policy_path = os.path.join(to_directory, "test_policy")
        with open(train_policy_path, "wb") as f, open(test_policy_path, "wb") as g:
            pickle.dump(self.train_policy, f)
            pickle.dump(self.test_policy, g)

    def load(self, from_directory: str):
        self.qnetwork.load_state_dict(torch.load(f"{from_directory}/qnetwork.weights"))
        train_policy_path = os.path.join(from_directory, "train_policy")
        test_policy_path = os.path.join(from_directory, "test_policy")
        with open(train_policy_path, "rb") as f, open(test_policy_path, "rb") as g:
            self.train_policy = pickle.load(f)
            self.test_policy = pickle.load(g)
        self.policy = self.train_policy

    def randomize(self):
        self.qnetwork.randomize()

    def to(self, device: torch.device):
        self.qnetwork.to(device)
        self.device = device


class DQN(IDQN[LinearNN]):
    def compute_qvalues(self, obs: Observation):
        obs_data = torch.from_numpy(obs.data).unsqueeze(0).to(self.device, non_blocking=True)
        obs_extras = torch.from_numpy(obs.extras).unsqueeze(0).to(self.device, non_blocking=True)
        return self.qnetwork.forward(obs_data, obs_extras).squeeze(0)


class RDQN(IDQN[RecurrentNN]):
    """
    Recurrent DQN

    Essentially the same as DQN, but we have to manage the hidden states.
    """

    def __init__(self, qnetwork: RecurrentNN, train_policy: Policy, test_policy: Policy | None = None):
        super().__init__(qnetwork, train_policy, test_policy)
        self._hidden_states = None
        self._saved_train_hidden_states = None

    def compute_qvalues(self, obs: Observation):
        obs_data = torch.from_numpy(obs.data).unsqueeze(0).to(self.device, non_blocking=True)
        obs_extras = torch.from_numpy(obs.extras).unsqueeze(0).to(self.device, non_blocking=True)
        qvalues, self._hidden_states = self.qnetwork.forward(obs_data, obs_extras, self._hidden_states)
        return qvalues.squeeze(0)

    def new_episode(self):
        self._hidden_states = None

    def set_testing(self):
        super().set_testing()
        self._saved_train_hidden_states = self._hidden_states

    def set_training(self):
        super().set_training()
        self._hidden_states = self._saved_train_hidden_states
