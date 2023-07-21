import os
from typing import Optional
from dataclasses import dataclass
from copy import deepcopy
import torch
import numpy as np
from rlenv.models import Transition, Observation
from marl import nn
from marl.models import  Batch, RLAlgo
from marl.logging import Logger
from marl.policy import Policy, EpsilonGreedy
from marl.utils import defaults_to

from .trainer import DQNTrainer

@dataclass
class DQN(RLAlgo):
    """
    Independent Deep Q-Network agent with shared QNetwork.
    If agents require different behaviours, an agentID should be included in the 
    observation 'extras'.
    """
    _qnetwork: nn.LinearNN
    _policy: Policy
    _device: torch.device
    _trainer: DQNTrainer

    def __init__(
        self,
        qnetwork: nn.LinearNN,
        trainer: DQNTrainer,
        train_policy: Optional[Policy]=None,
        test_policy: Optional[Policy]=None,
        logger: Optional[Logger]=None
    ):
        super().__init__(logger)
        self._trainer = trainer
        self._qnetwork = qnetwork
        self._qtarget = deepcopy(self._qnetwork)
        self._train_policy = defaults_to(train_policy, lambda: EpsilonGreedy.constant(0.1))
        self._test_policy = defaults_to(test_policy, lambda: EpsilonGreedy.constant(0.05))
        self._policy = self._train_policy
        self._device = self._qnetwork.device
        self._train_logs = {}

    @property
    def policy(self) -> Policy:
        return self._policy

    @torch.no_grad()
    def choose_action(self, obs: Observation) -> np.ndarray:
        qvalues = self.compute_qvalues(obs)
        qvalues = qvalues.cpu().numpy()
        return self._policy.get_action(qvalues, obs.available_actions)
    
    @torch.no_grad()
    def value(self, obs: Observation) -> float:
        qvalues = self.compute_qvalues(obs)
        max_qvalues = torch.max(qvalues, dim=-1).values
        return torch.mean(max_qvalues).item()
    
    def after_train_step(self, transition: Transition, time_step: int):
        self._trainer.update(transition, time_step)
    
    def compute_qvalues(self, data: Batch|Observation) -> torch.Tensor:
        match data:
            case Batch() as batch:
                qvalues = self._qnetwork.forward(batch.obs, batch.extras)
                qvalues = torch.gather(qvalues, index=batch.actions, dim=-1)
                return qvalues.squeeze(dim=-1)
            case Observation() as obs:
                obs_data = torch.from_numpy(obs.data).to(self._device, non_blocking=True)
                obs_extras = torch.from_numpy(obs.extras).to(self._device, non_blocking=True)
                return self._qnetwork.forward(obs_data, obs_extras)
            case _: raise ValueError("Invalid input data type for 'compute_qvalues'")

    def before_tests(self, time_step: int):
        self._policy = self._test_policy

    def after_tests(self, time_step: int, episodes):
        self._policy = self._train_policy

    def save(self, to_directory: str):
        os.makedirs(to_directory, exist_ok=True)
        qnetwork_path = os.path.join(to_directory, "qnetwork.weights")
        qtarget_path = os.path.join(to_directory, "qtarget.weights")
        train_policy_path = os.path.join(to_directory, "train_policy")
        test_policy_path = os.path.join(to_directory, "test_policy")
        torch.save(self._qnetwork.state_dict(), qnetwork_path)
        torch.save(self._qtarget.state_dict(), qtarget_path)
        self._train_policy.save(train_policy_path)
        self._test_policy.save(test_policy_path)
        

    def load(self, from_directory: str):
        qnetwork_path = os.path.join(from_directory, "qnetwork.weights")
        qtarget_path = os.path.join(from_directory, "qtarget.weights")
        train_policy_path = os.path.join(from_directory, "train_policy")
        test_policy_path = os.path.join(from_directory, "test_policy")
        self._qnetwork.load_state_dict(torch.load(qnetwork_path))
        self._qtarget.load_state_dict(torch.load(qtarget_path))
        self._train_policy = self._train_policy.__class__.load(train_policy_path)
        self._test_policy = self._test_policy.__class__.load(test_policy_path)
        self._policy = self._train_policy

    def to(self, device: torch.device):
        self._qnetwork.to(device)
        self._qtarget.to(device)
        self._device = device

