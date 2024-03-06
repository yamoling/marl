import os
import pickle
from dataclasses import dataclass
from typing import Optional
from serde import serde

import numpy as np
import torch
from rlenv.models import Observation

from marl.models import RLAlgo, Policy, QNetwork, RecurrentQNetwork


@serde
@dataclass
class DQN(RLAlgo):
    """
    Deep Q-Network Interface with shared QNetwork.
    """

    qnetwork: QNetwork
    train_policy: Policy
    test_policy: Policy

    def __init__(self, qnetwork: QNetwork, train_policy: Policy, test_policy: Optional[Policy] = None):
        super().__init__()
        self.qnetwork = qnetwork
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
        return self.qnetwork.value(obs).item()

    def compute_qvalues(self, obs: Observation) -> torch.Tensor:
        return self.qnetwork.qvalues(obs)

    def set_testing(self):
        self.policy = self.test_policy
        self.qnetwork.eval()

    def set_training(self):
        self.policy = self.train_policy
        self.qnetwork.train()

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


class RDQN(DQN):
    """
    Recurrent DQN.

    Essentially the same as DQN, but we have to tell the q-network to reset hidden states at each new episode.
    """

    def __init__(self, qnetwork: RecurrentQNetwork, train_policy: Policy, test_policy: Policy | None = None):
        super().__init__(qnetwork, train_policy, test_policy)
        self.qnetwork: RecurrentQNetwork

    def new_episode(self):
        self.qnetwork.reset_hidden_states()
