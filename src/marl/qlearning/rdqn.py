from dataclasses import dataclass
from typing import Any, Optional
import numpy as np
import numpy.typing as npt
from rlenv.models import Observation
import torch
from rlenv import Observation, Episode
from marl import nn
from marl.models import RLAlgo
from marl.policy import Policy


@dataclass
class RDQN(RLAlgo):
    qnetwork: nn.RecurrentNN

    def __init__(self, qnetwork: nn.RecurrentNN, train_policy: Policy, test_policy: Optional[Policy] = None):
        super().__init__()
        self.qnetwork = qnetwork
        self.device = self.qnetwork.device
        self.train_policy = train_policy
        if test_policy is None:
            test_policy = self.train_policy
        self.test_policy = test_policy
        self._hidden_states = None

    def choose_action(self, observation: Observation):
        with torch.no_grad():
            qvalues = self.compute_qvalues(observation)
        return super().choose_action(observation)

    def compute_qvalues(self, obs: Observation) -> torch.Tensor:
        obs_data = torch.from_numpy(obs.data).to(self.device, non_blocking=True)
        obs_extras = torch.from_numpy(obs.extras).to(self.device, non_blocking=True)
        qvalues, self._hidden_states = self.qnetwork.forward(obs_data, obs_extras)
        return qvalues

    def set_testing(self):
        self.policy = self.test_policy

    def set_training(self):
        self.policy = self.train_policy
