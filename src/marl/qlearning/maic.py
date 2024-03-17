import os
import pickle
import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional
from rlenv.models import Observation
from marl.models import RLAlgo, Policy, NN, MAICNN


@dataclass
class MAICAlgo(RLAlgo):
    maic_network: MAICNN
    train_policy: Policy
    test_policy: Policy

    def __init__(self, maic_network: MAICNN, train_policy: Policy, test_policy: Policy, args):
        super().__init__()
        self.maic_network = maic_network
        self.n_agents = args.n_agents
        self.args = args
        self.train_policy = train_policy
        if test_policy is None:
            test_policy = self.train_policy
        self.test_policy = test_policy
        self.policy = self.train_policy
        
        self.hidden_states = None

    def to_tensor(self, obs: Observation) -> tuple[torch.Tensor, torch.Tensor]:
        extras = torch.from_numpy(obs.extras).unsqueeze(0).to(self.device)
        obs_tensor = torch.from_numpy(obs.data).unsqueeze(0).to(self.device)
        return obs_tensor, extras
    
    def choose_action(self, obs: Observation) -> np.ndarray[np.int32, Any]:
        with torch.no_grad():
            qvalues, _ = self.forward(*self.to_tensor(obs), test_mode=False)
        qvalues = qvalues.cpu().numpy()
        return self.policy.get_action(qvalues, obs.available_actions)
    
    def value(self, obs: Observation) -> float:
        """Get the value of the input observation"""
        return self.maic_network.value(obs, self.hidden_states).item()
        
    def forward(self, obs: torch.Tensor, extras: torch.Tensor, test_mode=False):

        agent_outs, self.hidden_states, losses = self.maic_network.qvalues(obs, extras, self.hidden_states, 
            test_mode=test_mode)

        return agent_outs, losses # bs in the view


    def init_hidden(self, batch_size):
        self.hidden_states = self.maic_network.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)

    def new_episode(self):
        self.init_hidden(1)  

    def parameters(self):
        return self.maic_network.parameters()

    def set_testing(self):
        self.policy = self.test_policy
        self.maic_network.eval()

    def set_training(self):
        self.policy = self.train_policy
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
        self.maic_network.load_state_dict(torch.load(f"{from_directory}/maic_network.weights"))
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