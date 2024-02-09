import os
from typing import Optional
from dataclasses import dataclass
from copy import deepcopy
import torch
import numpy as np
from rlenv.models import Transition, Observation, Metrics
from marl import nn
from marl.models import ReplayMemory, TransitionMemory, Batch, TransitionBatch
from marl.policy import Policy, EpsilonGreedy
from marl.utils import defaults_to, get_device
from marl.logging import Logger

from .qlearning import IDeepQLearning


@dataclass
class DQN(IDeepQLearning):
    """
    Independent Deep Q-Network agent with shared QNetwork.
    If agents require different behaviours, an agentID should be included in the
    observation 'extras'.
    """

    _gamma: float
    _tau: float
    _batch_size: int
    _qnetwork: nn.LinearNN
    _optimizer: torch.optim.Optimizer
    _policy: Policy
    _memory: TransitionMemory
    _device: torch.device
    _qnetwork: nn.LinearNN
    _qtarget: nn.LinearNN

    def __init__(
        self,
        qnetwork: nn.LinearNN,
        gamma=0.99,
        tau=1e-2,
        batch_size=64,
        lr=1e-4,
        optimizer: Optional[torch.optim.Optimizer] = None,
        train_policy: Optional[Policy] = None,
        test_policy: Optional[Policy] = None,
        memory: Optional[TransitionMemory] = None,
        device: Optional[torch.device] = None,
        update_frequency=200,
        use_soft_update=True,
        double_qlearning=True,
        logger: Optional[Logger] = None,
        train_interval=1,
    ):
        """Soft update tau value"""
        super().__init__(logger)
        self._gamma = gamma
        self._tau = tau
        self._batch_size = batch_size
        self._device = defaults_to(device, get_device)
        self._qnetwork = qnetwork.to(self._device, non_blocking=True)
        self._qtarget = deepcopy(self._qnetwork).to(self._device, non_blocking=True)
        self._memory = defaults_to(memory, lambda: TransitionMemory(50_000))
        self._optimizer = defaults_to(optimizer, lambda: torch.optim.Adam(qnetwork.parameters(), lr=lr))
        self._train_policy = defaults_to(train_policy, lambda: EpsilonGreedy.constant(0.1))
        self._test_policy = defaults_to(test_policy, lambda: EpsilonGreedy.constant(0.05))
        self._policy = self._train_policy
        self._parameters = list(self._qnetwork.parameters())
        self._double_qlearning = double_qlearning
        self._update_frequency = update_frequency
        self._use_soft_update = use_soft_update
        self._train_interval = train_interval
        self._train_logs = {}

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def memory(self) -> ReplayMemory[Transition, TransitionBatch]:
        return self._memory

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
        self._memory.add(transition)
        self.update(time_step)

    def compute_qvalues(self, data: Batch | Observation) -> torch.Tensor:
        match data:
            case Batch() as batch:
                qvalues = self._qnetwork.forward(batch.obs, batch.extras)
                qvalues = torch.gather(qvalues, index=batch.actions, dim=-1)
                return qvalues.squeeze(dim=-1)
            case Observation() as obs:
                obs_data = torch.from_numpy(obs.data).to(self._device, non_blocking=True)
                obs_extras = torch.from_numpy(obs.extras).to(self._device, non_blocking=True)
                return self._qnetwork.forward(obs_data, obs_extras)
            case _:
                raise ValueError("Invalid input data type for 'compute_qvalues'")

    def double_qlearning(self, batch: Batch) -> torch.Tensor:
        # 1) Take the max qvalues from the online network
        next_qvalues = self._qnetwork.forward(batch.obs_, batch.extras_)
        next_qvalues[batch.available_actions_ == 0.0] = -torch.inf
        indices = torch.unsqueeze(next_qvalues.max(dim=-1)[1], -1)
        # 2) Take the qvalues from the target network
        next_qvalues = self._qtarget.forward(batch.obs_, batch.extras_)
        next_qvalues = next_qvalues.gather(-1, indices).squeeze(-1)
        return next_qvalues

    @torch.no_grad()
    def compute_targets(self, batch: Batch) -> torch.Tensor:
        if self._double_qlearning:
            next_qvalues = self.double_qlearning(batch)
        else:
            next_qvalues = self._qtarget.forward(batch.obs_, batch.extras_)
            next_qvalues[batch.available_actions_ == 0.0] = -torch.inf
            next_qvalues = torch.max(next_qvalues, dim=-1)[0]
        target_qvalues = batch.rewards + self._gamma * next_qvalues * (1 - batch.dones)
        return target_qvalues

    def compute_loss(self, qvalues: torch.Tensor, qtargets: torch.Tensor, batch: Batch) -> torch.Tensor:
        # Mean squared error
        mse = (qvalues - qtargets) ** 2
        # Apply importance sampling weights is necessary
        if batch.importance_sampling_weights is not None:
            mse = mse * batch.importance_sampling_weights
        return torch.mean(mse)

    def process_batch(self, batch: Batch) -> Batch:
        return batch.for_individual_learners()

    def update(self, update_step: int):
        if len(self._memory) < self._batch_size or update_step % self._train_interval != 0:
            return
        batch = self._memory.sample(self._batch_size).to(self._device)
        batch = self.process_batch(batch)
        # Compute qvalues and qtargets (delegated to child classes)
        qvalues = self.compute_qvalues(batch)
        qtargets = self.compute_targets(batch).detach()
        assert (
            qvalues.shape == qtargets.shape
        ), f"Predicted qvalues ({qvalues.shape}) and target qvalues ({qtargets.shape}) do not have the same shape !"
        # Compute the loss and apply gradient descent
        loss = self.compute_loss(qvalues, qtargets, batch)
        self._optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self._parameters, 10)
        self._optimizer.step()

        self._train_policy.update()
        if self._use_soft_update:
            self._target_soft_update()
        else:
            self._target_update(update_step)
        self._memory.update(batch, qvalues, qtargets)

        if self.logger is not None:
            logs = Metrics(
                **self._train_logs,
                loss=loss.item(),
                grad_norm=grad_norm.item(),
                epsilon=self._train_policy._epsilon.value,
            )
            self.logger.log("training_data", logs, update_step)

    def _target_soft_update(self):
        for param, target_param in zip(self._qnetwork.parameters(), self._qtarget.parameters()):
            new_value = (1 - self._tau) * target_param.data + self._tau * param.data
            target_param.data.copy_(new_value, non_blocking=True)

    def _target_update(self, time_step: int):
        if time_step % self._update_frequency == 0:
            self._qtarget.load_state_dict(self._qnetwork.state_dict())

    def before_tests(self, time_step: int):
        self._policy = self._test_policy
        return super().before_tests(time_step)

    def after_tests(self, time_step: int, episodes):
        self._policy = self._train_policy
        return super().after_tests(time_step, episodes)

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

    def summary(self) -> dict[str,]:
        return {
            "name": self.__class__.__name__,
            "gamma": self._gamma,
            "batch_size": self._batch_size,
            "tau": self._tau,
            "optimizer": {
                "name": self._optimizer.__class__.__name__,
                "learning rate": self._optimizer.param_groups[0]["lr"],
            },
            "memory": self._memory.summary(),
            "qnetwork": self._qnetwork.summary(),
            "train_policy": self._train_policy.summary(),
            "test_policy": self._test_policy.summary(),
            "use_soft_update": self._use_soft_update,
            "update_frequency": self._update_frequency,
            "train_interval": self._train_interval,
            "double_qlearning": self._double_qlearning,
        }

    @classmethod
    def from_summary(cls, summary: dict[str,]):
        device = defaults_to(summary.get("device"), get_device)
        summary["device"] = device
        from marl import policy

        summary["train_policy"] = policy.from_summary(summary["train_policy"])
        summary["test_policy"] = policy.from_summary(summary["test_policy"])
        from marl import nn

        summary["qnetwork"] = nn.from_summary(summary["qnetwork"])
        from marl.models import replay_memory

        summary["memory"] = replay_memory.from_summary(summary["memory"])
        summary["lr"] = summary.pop("optimizer")["learning rate"]
        return super().from_summary(summary)
