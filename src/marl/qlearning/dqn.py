import os
from dataclasses import dataclass
from copy import deepcopy
import torch
from rlenv import Transition, Observation
from marl import nn
from marl.models import TransitionMemory, Batch
from marl.policy import Policy, EpsilonGreedy
from marl.utils import defaults_to, get_device

from .qlearning import IDeepQLearning

@dataclass
class DQN(IDeepQLearning):
    """
    Independent Deep Q-Network agent with shared QNetwork.
    If agents require different behaviours, an agentID should be included in the 
    observation 'extras'.
    """
    gamma: float
    tau: float
    batch_size: int
    lr: float
    qnetwork: nn.LinearNN
    optimizer: torch.optim.Optimizer
    policy: Policy
    memory: TransitionMemory
    device: torch.device

    def __init__(
        self,
        qnetwork: nn.LinearNN,
        gamma=0.99,
        tau=1e-2,
        batch_size=64,
        lr=1e-4,
        optimizer: torch.optim.Optimizer=None,
        train_policy: Policy=None,
        test_policy: Policy=None,
        memory: TransitionMemory=None,
        device: torch.device=None,
    ):
        """Soft update tau value"""
        super().__init__()
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = defaults_to(device, get_device())
        self.qnetwork = qnetwork.to(self.device)
        self.qtarget = deepcopy(self.qnetwork).to(self.device)
        self.loss_function = torch.nn.MSELoss()
        self.memory = defaults_to(memory, TransitionMemory(50_000))
        self.optimizer = defaults_to(optimizer, torch.optim.Adam(qnetwork.parameters(), lr=lr))
        self.train_policy = defaults_to(train_policy, EpsilonGreedy(0.1))
        self.test_policy = defaults_to(test_policy, EpsilonGreedy(0.01))
        self.policy = self.train_policy

    def choose_action(self, obs: Observation) -> list[int]:
        with torch.no_grad():
            qvalues = self.compute_qvalues(obs)
            qvalues = qvalues.cpu().numpy()
        return self.policy.get_action(qvalues, obs.available_actions)
    
    def after_step(self, transition: Transition, _step_num: int):
        self.memory.add(transition)
        self.update()
    
    def compute_qvalues(self, data: Batch|Observation) -> torch.Tensor:
        match data:
            case Batch() as batch:
                qvalues = self.qnetwork.forward(batch.obs, batch.extras)
                qvalues = qvalues.gather(index=batch.actions, dim=-1)
                return qvalues.squeeze(dim=-1)
            case Observation() as obs:
                obs_data = torch.from_numpy(obs.data).to(self.device, non_blocking=True)
                obs_extras = torch.from_numpy(obs.extras).to(self.device, non_blocking=True)
                return self.qnetwork.forward(obs_data, obs_extras)
            case _: raise ValueError("Invalid input data type for 'compute_qvalues'")

    def compute_targets(self, batch: Batch) -> torch.Tensor:
        next_qvalues = self.qtarget.forward(batch.obs_, batch.extras_)
        next_qvalues[batch.available_actions_ == 0.0] = -torch.inf
        next_qvalues = torch.max(next_qvalues, dim=-1)[0]
        target_qvalues = batch.rewards + self.gamma * next_qvalues * (1 - batch.dones)
        return target_qvalues

    def compute_loss(self, qvalues: torch.Tensor, qtargets: torch.Tensor, batch: Batch) -> torch.Tensor:
        self.memory.update(batch.sample_index, qvalues, qtargets)
        return self.loss_function(qvalues, qtargets)

    def process_batch(self, batch: Batch) -> Batch:
        return batch.for_individual_learners()

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        batch = self.memory.sample(self.batch_size).to(self.device)
        batch = self.process_batch(batch)
        # Compute qvalues and qtargets (delegated to child classes)
        qvalues = self.compute_qvalues(batch)
        with torch.no_grad():
            qtargets = self.compute_targets(batch)
        assert qvalues.shape == qtargets.shape, f"Predicted qvalues ({qvalues.shape}) and target qvalues ({qtargets.shape}) do not have the same shape !"
        # Compute the loss and apply gradient descent
        loss = self.compute_loss(qvalues, qtargets, batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step(None)
        self.train_policy.update()
        self._target_soft_update()

    def _target_soft_update(self):
        for param, target_param in zip(self.qnetwork.parameters(), self.qtarget.parameters()):
            new_value = (1-self.tau) * target_param.data + self.tau * param.data
            target_param.data.copy_(new_value, non_blocking=True)

    def before_tests(self):
        self.policy = self.test_policy
        return super().before_tests()

    def after_tests(self, time_step: int, episodes):
        self.policy = self.train_policy
        return super().after_tests(time_step, episodes)

    def save(self, to_path: str):
        os.makedirs(to_path, exist_ok=True)
        qnetwork_path = os.path.join(to_path, "qnetwork.weights")
        qtarget_path = os.path.join(to_path, "qtarget.weights")
        train_policy_path = os.path.join(to_path, "train_policy")
        test_policy_path = os.path.join(to_path, "test_policy")
        torch.save(self.qnetwork.state_dict(), qnetwork_path)
        torch.save(self.qtarget.state_dict(), qtarget_path)
        self.train_policy.save(train_policy_path)
        self.test_policy.save(test_policy_path)
        

    def load(self, from_path: str):
        qnetwork_path = os.path.join(from_path, "qnetwork.weights")
        qtarget_path = os.path.join(from_path, "qtarget.weights")
        train_policy_path = os.path.join(from_path, "train_policy")
        test_policy_path = os.path.join(from_path, "test_policy")
        self.qnetwork.load_state_dict(torch.load(qnetwork_path))
        self.qtarget.load_state_dict(torch.load(qtarget_path))
        self.train_policy.load(train_policy_path)
        self.test_policy.load(test_policy_path)

    def summary(self) -> dict[str,]:
        return {
            "name": self.__class__.__name__,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "optimizer": {
                "name": self.optimizer.__class__.__name__,
                "learning rate": self.optimizer.param_groups[0]["lr"]
            },
            "memory": {
                "type": self.memory.__class__.__name__,
                "size": len(self.memory)
            },
            "qnetwork": str(self.qnetwork),
            "train_policy": {
                "name": self.train_policy.__class__.__name__,
                **self.train_policy.__dict__
            },
            "test_policy" : {
                "name": self.test_policy.__class__.__name__,
                **self.test_policy.__dict__
            }
        }