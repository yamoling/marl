import os
import pickle
from copy import deepcopy
import numpy as np
import torch
from rlenv.models import Observation

from typing import Optional

from marl.models import RLAlgo, Policy, QNetwork, CommNetwork


class RIALAlgo(RLAlgo):
    def __init__(self, comm_network: CommNetwork, qnetwork: QNetwork, train_policy: Policy, test_policy: Optional[Policy] = None):
        super().__init__()
        self.comm_network = comm_network
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
        message = self._encode_message(obs)
        q_input = self._add_message_to_observation(obs, message)
        return self.qnetwork.value(q_input).item()

    def compute_qvalues(self, obs: Observation) -> torch.Tensor:
        message = self._encode_message(obs)
        q_input = self._add_message_to_observation(obs, message)
        return self.qnetwork.qvalues(q_input)

    def _encode_message(self, obs: Observation) -> torch.Tensor:
        return self.comm_network.encode(obs)

    def _add_message_to_observation(self, obs: Observation, message: torch.Tensor) -> Observation:
        obs_copy = deepcopy(obs)
        obs_copy.extras = np.concatenate([obs.extras, message.detach().cpu().numpy()], axis=-1)
        return obs_copy

    def set_testing(self):
        self.policy = self.test_policy
        self.qnetwork.eval()

    def set_training(self):
        self.policy = self.train_policy
        self.qnetwork.train()
    
    #TODO : Save CommNetwork weights
    def save(self, to_directory: str):
        os.makedirs(to_directory, exist_ok=True)
        torch.save(self.qnetwork.state_dict(), f"{to_directory}/qnetwork.weights")
        train_policy_path = os.path.join(to_directory, "train_policy")
        test_policy_path = os.path.join(to_directory, "test_policy")
        with open(train_policy_path, "wb") as f, open(test_policy_path, "wb") as g:
            pickle.dump(self.train_policy, f)
            pickle.dump(self.test_policy, g)
    #TODO : Load CommNetwork weights
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
        self.comm_network.randomize()
    
    def to(self, device: torch.device):
        self.qnetwork.to(device)
        self.comm_network.to(device)

        