import os
import pickle
from dataclasses import dataclass
from typing import Optional

import torch
from marlenv import Observation

from marl.models import Policy, QNetwork, RecurrentQNetwork

from ..agent import Agent


@dataclass
class DQN(Agent):
    """
    Deep Q-Network Interface with shared QNetwork.
    """

    qnetwork: QNetwork
    train_policy: Policy
    test_policy: Policy

    def __init__(
        self,
        qnetwork: QNetwork,
        train_policy: Policy,
        test_policy: Optional[Policy] = None,
    ):
        super().__init__()
        self.qnetwork = qnetwork
        self.train_policy = train_policy
        if test_policy is None:
            test_policy = self.train_policy
        self.test_policy = test_policy
        self.policy = self.train_policy

    def choose_action(self, obs: Observation):
        with torch.no_grad():
            qvalues = self.qnetwork.qvalues(obs)
        qvalues = qvalues.numpy(force=True)
        return self.policy.get_action(qvalues, obs.available_actions)

    def value(self, obs: Observation) -> float:
        return self.qnetwork.value(obs).item()

    def set_testing(self):
        self.policy = self.test_policy
        self.qnetwork.set_testing(True)
        self.qnetwork.eval()

    def set_training(self):
        self.policy = self.train_policy
        self.qnetwork.set_testing(False)
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
        self.qnetwork.load_state_dict(torch.load(f"{from_directory}/qnetwork.weights", weights_only=True, map_location="cpu"))
        train_policy_path = os.path.join(from_directory, "train_policy")
        test_policy_path = os.path.join(from_directory, "test_policy")
        with open(train_policy_path, "rb") as f, open(test_policy_path, "rb") as g:
            self.train_policy = pickle.load(f)
            self.test_policy = pickle.load(g)
        self.policy = self.train_policy


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
