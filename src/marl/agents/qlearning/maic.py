import os
import pickle
import torch
from dataclasses import dataclass
from marlenv.models import Observation
from marl.models import Policy, MAICNN

import numpy as np
from ..agent import Agent


@dataclass
class MAICParameters:
    n_agents: int
    latent_dim: int = 8
    nn_hidden_size: int = 64
    rnn_hidden_dim: int = 64
    attention_dim: int = 32
    var_floor: float = 0.002
    mi_loss_weight: float = 0.001
    entropy_loss_weight: float = 0.01
    com: bool = True


@dataclass
class MAIC(Agent):
    maic_network: MAICNN
    train_policy: Policy
    test_policy: Policy

    def __init__(self, maic_network: MAICNN, train_policy: Policy, test_policy: Policy, args: MAICParameters):
        super().__init__(0)
        self.maic_network = maic_network
        self.n_agents = args.n_agents
        self.args = args
        self.train_policy = train_policy
        if test_policy is None:
            test_policy = self.train_policy
        self.test_policy = test_policy
        self.policy = self.train_policy
        self.test_mode = True

        self.hidden_states = None

    def to_tensor(self, obs: Observation) -> tuple[torch.Tensor, torch.Tensor]:
        extras = torch.from_numpy(obs.extras).unsqueeze(0).to(self.device)
        obs_tensor = torch.from_numpy(obs.data).unsqueeze(0).to(self.device)
        return obs_tensor, extras

    def choose_action(self, obs: Observation) -> np.ndarray:
        with torch.no_grad():
            qvalues = self.compute_qvalues(obs)
        qvalues = qvalues.cpu().numpy()
        return self.policy.get_action(qvalues, obs.available_actions)

    def value(self, obs: Observation) -> float:
        """Get the value of the input observation"""
        return self.maic_network.value(obs).item()

    def compute_qvalues(self, obs: Observation) -> torch.Tensor:
        objective_qvalues = self.maic_network.qvalues(obs)
        return torch.sum(objective_qvalues, dim=-1)

    def new_episode(self):
        self.maic_network.reset_hidden_states()

    def set_testing(self):
        self.policy = self.test_policy
        self.maic_network.set_testing(True)
        self.maic_network.eval()

    def set_training(self):
        self.policy = self.train_policy
        self.maic_network.set_testing(False)
        self.maic_network.train()

    def save(self, to_directory: str):
        os.makedirs(to_directory, exist_ok=True)
        torch.save(self.maic_network.state_dict(), f"{to_directory}/maic_network.weights")
        train_policy_path = os.path.join(to_directory, "train_policy")
        test_policy_path = os.path.join(to_directory, "test_policy")
        with open(train_policy_path, "wb") as f, open(test_policy_path, "wb") as g:
            pickle.dump(self.train_policy, f)
            pickle.dump(self.test_policy, g)

    def load(self, from_directory: str):
        self.maic_network.load_state_dict(torch.load(f"{from_directory}/maic_network.weights", weights_only=True))
        train_policy_path = os.path.join(from_directory, "train_policy")
        test_policy_path = os.path.join(from_directory, "test_policy")
        with open(train_policy_path, "rb") as f, open(test_policy_path, "rb") as g:
            self.train_policy = pickle.load(f)
            self.test_policy = pickle.load(g)
        self.policy = self.train_policy

    def randomize(self):
        self.maic_network.randomize()

    def to(self, device: torch.device):
        self.maic_network.to(device)
        self.device = device
