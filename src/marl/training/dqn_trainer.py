import torch
import torch.nn.functional as F

from typing import Any, Literal, Optional
from copy import deepcopy
from rlenv import Transition, Episode
from marl.models import QNetwork, Mixer, ReplayMemory, Trainer, Policy, PrioritizedMemory
from marl.models.batch import EpisodeBatch, Batch
from marl.intrinsic_reward import IRModule
from .qtarget_updater import TargetParametersUpdater, SoftUpdate
from marl.utils import defaults_to

from dataclasses import dataclass
from serde import serialize


@serialize
@dataclass
class DQNTrainer(Trainer):
    qnetwork: QNetwork
    policy: Policy
    memory: ReplayMemory
    gamma: float
    batch_size: int
    target_updater: TargetParametersUpdater
    double_qlearning: bool
    mixer: Mixer
    ir_module: Optional[IRModule]
    grad_norm_clipping: Optional[float]

    def __init__(
        self,
        qnetwork: QNetwork,
        train_policy: Policy,
        memory: ReplayMemory,
        mixer: Mixer,
        gamma: float = 0.99,
        batch_size: int = 64,
        lr: float = 1e-4,
        optimiser: Literal["adam", "rmsprop"] = "adam",
        target_updater: Optional[TargetParametersUpdater] = None,
        double_qlearning: bool = False,
        train_interval: tuple[int, Literal["step", "episode"]] = (5, "step"),
        ir_module: Optional[IRModule] = None,
        grad_norm_clipping: Optional[float] = None,
    ):
        super().__init__(train_interval[1], train_interval[0])
        self.qnetwork = qnetwork
        self.qtarget = deepcopy(qnetwork)
        self.device = qnetwork.device
        self.policy = train_policy
        self.memory = memory
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_updater = defaults_to(target_updater, lambda: SoftUpdate(1e-2))
        self.double_qlearning = double_qlearning
        self.mixer = mixer
        self.target_mixer = deepcopy(mixer)
        self.ir_module = ir_module
        self.update_num = 0

        # Parameters and optimiser
        self.grad_norm_clipping = grad_norm_clipping
        self.target_updater.add_parameters(qnetwork.parameters(), self.qtarget.parameters())
        self.target_updater.add_parameters(mixer.parameters(), self.target_mixer.parameters())
        match optimiser:
            case "adam":
                self.optimiser = torch.optim.Adam(self.target_updater.parameters, lr=lr)
            case "rmsprop":
                self.optimiser = torch.optim.RMSprop(self.target_updater.parameters, lr=lr, eps=1e-5)
            case other:
                raise ValueError(f"Unknown optimiser: {other}. Expected 'adam' or 'rmsprop'.")

    def _update(self, time_step: int):
        self.update_num += 1
        if self.update_num % self.update_interval != 0 or not self._can_update():
            return {}
        logs, td_error = self.optimise_qnetwork()
        logs = logs | self.policy.update(time_step)
        logs = logs | self.target_updater.update(time_step)
        if isinstance(self.memory, PrioritizedMemory):
            logs = logs | self.memory.update(td_error)
        if self.ir_module is not None:
            logs = logs | self.ir_module.update(time_step)
        return logs

    def _can_update(self):
        return self.memory.can_sample(self.batch_size)

    def _next_state_value(self, batch: Batch):
        """this works"""
        next_qvalues = self.qtarget.batch_forward(batch.obs_, batch.extras_)
        # next_qvalues = self.qtarget.batch_forward(batch.all_obs, batch.all_extras)[1:]
        # For double q-learning, we use the qnetwork to select the best action. Otherwise, we use the target qnetwork.
        if self.double_qlearning:
            qvalues_for_index = self.qnetwork.batch_forward(batch.obs_, batch.extras_)
            # qvalues_for_index = self.qnetwork.batch_forward(batch.all_obs, batch.all_extras)[1:]
        else:
            qvalues_for_index = next_qvalues
        # For episode batches, the batch includes the initial observation in order to compute the
        # hidden state at t=0 and use it for t=1. We need to remove it when considering the next qvalues.
        if isinstance(batch, EpisodeBatch):
            next_qvalues = next_qvalues[1:]
            qvalues_for_index = qvalues_for_index[1:]
        qvalues_for_index = torch.sum(qvalues_for_index, -1)
        qvalues_for_index[batch.available_actions_ == 0.0] = -torch.inf
        indices = torch.argmax(qvalues_for_index, dim=-1, keepdim=True)
        indices = indices.unsqueeze(-1).repeat(*(1 for _ in indices.shape), batch.reward_size)
        next_values = torch.gather(next_qvalues, -2, indices).squeeze(-2)
        if self.target_mixer is not None:
            next_values = self.target_mixer.forward(next_values, batch.states_, batch.one_hot_actions, next_qvalues)
        return next_values

    def _next_state_value2(self, batch: Batch):
        next_qvalues = self.qtarget.batch_forward(batch.all_obs, batch.all_extras)[1:]
        # For double q-learning, we use the qnetwork to select the best action. Otherwise, we use the target qnetwork.
        if self.double_qlearning:
            qvalues_for_index = self.qnetwork.batch_forward(batch.all_obs, batch.all_extras)[1:]
        else:
            qvalues_for_index = next_qvalues
        qvalues_for_index = torch.sum(qvalues_for_index, -1)
        qvalues_for_index[batch.available_actions_ == 0.0] = -torch.inf
        indices = torch.argmax(qvalues_for_index, dim=-1, keepdim=True)
        indices = indices.unsqueeze(-1).repeat(*(1 for _ in indices.shape), batch.reward_size)
        next_values = torch.gather(next_qvalues, -2, indices).squeeze(-2)
        if self.target_mixer is not None:
            next_values = self.target_mixer.forward(next_values, batch.states_, batch.one_hot_actions, next_qvalues)
        return next_values

    def optimise_qnetwork(self):
        batch = self.memory.sample(self.batch_size).to(self.qnetwork.device)
        batch.multi_objective()
        if self.ir_module is not None:
            batch.rewards = batch.rewards + self.ir_module.compute(batch)

        # Qvalues computation
        all_qvalues = self.qnetwork.batch_forward(batch.obs, batch.extras)
        qvalues = torch.gather(all_qvalues, dim=-2, index=batch.actions).squeeze(-2)
        if self.mixer is not None:
            qvalues = self.mixer.forward(qvalues, batch.states, batch.one_hot_actions, all_qvalues)

        # Qtargets computation
        next_values = self._next_state_value2(batch)
        assert batch.rewards.shape == next_values.shape == batch.dones.shape
        qtargets = batch.rewards + self.gamma * next_values * (1 - batch.dones)
        assert qvalues.shape == qtargets.shape
        # Compute the loss
        td_error = qvalues - qtargets.detach()
        td_error = td_error * batch.masks
        squared_error = td_error**2
        if batch.importance_sampling_weights is not None:
            assert squared_error.shape == batch.importance_sampling_weights.shape
            squared_error = squared_error * batch.importance_sampling_weights
        loss = squared_error.sum() / batch.masks.sum()

        # Optimize
        logs = {"loss": float(loss.item())}
        self.optimiser.zero_grad()
        loss.backward()
        if self.grad_norm_clipping is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.qnetwork.parameters(), self.grad_norm_clipping)
            logs["grad_norm"] = grad_norm.item()
        self.optimiser.step()
        return logs, td_error

    def _next_state_value_old(self, batch: Batch):
        all_target_qvalues = self.qtarget.batch_forward(batch.all_obs, batch.all_extras)
        target_qvalues_ = all_target_qvalues[1:]
        if self.double_qlearning:
            qvalues_ = self.qnetwork.batch_forward(batch.all_obs, batch.all_extras)[1:]
            qvalues_[batch.available_actions_ == 0] = -torch.inf
            next_qvalues = qvalues_
        else:
            target_qvalues_[batch.available_actions_ == 0] = -torch.inf
            next_qvalues = target_qvalues_
        # Weighted sum
        next_qvalues = next_qvalues.sum(dim=-1)
        indices = torch.argmax(next_qvalues, dim=-1, keepdim=True)
        indices = indices.unsqueeze(-1).repeat(*(1 for _ in indices.shape), batch.reward_size)
        next_values = torch.gather(target_qvalues_, -2, indices).squeeze(-2)
        next_values = self.target_mixer.forward(next_values, batch.states_, batch.one_hot_actions, target_qvalues_)
        return next_values

    def optimise_qnetwork_old(self):
        batch = self.memory.sample(self.batch_size).to(self.qnetwork.device)
        batch.multi_objective()
        # batch.rewards = batch.rewards.squeeze()
        if self.ir_module is not None:
            batch.rewards = batch.rewards + self.ir_module.compute(batch)

        # Qvalues computation
        qvalues = self.qnetwork.batch_forward(batch.obs, batch.extras)
        chosen_qvalues = torch.gather(qvalues, dim=-2, index=batch.actions).squeeze(-2)
        chosen_qvalues = self.mixer.forward(chosen_qvalues, batch.states, batch.one_hot_actions, qvalues)

        # Qtargets computation
        next_values = self._next_state_value(batch)
        assert batch.rewards.shape == next_values.shape == batch.dones.shape
        qtargets = batch.rewards + self.gamma * next_values * (1 - batch.dones)

        # Compute the loss
        assert chosen_qvalues.shape == qtargets.shape == batch.masks.shape
        td_error = chosen_qvalues - qtargets.detach()
        td_error = td_error * batch.masks
        squared_error = td_error**2
        if batch.importance_sampling_weights is not None:
            assert squared_error.shape == batch.importance_sampling_weights.shape
            squared_error = squared_error * batch.importance_sampling_weights
        loss = squared_error.sum() / batch.masks.sum()

        # Optimize
        logs = {"loss": float(loss.item())}
        self.optimiser.zero_grad()
        loss.backward()
        if self.grad_norm_clipping is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.qnetwork.parameters(), self.grad_norm_clipping)
            logs["grad_norm"] = grad_norm.item()
        self.optimiser.step()
        return logs, td_error

    def update_step(self, transition: Transition, time_step: int) -> dict[str, Any]:
        if self.memory.update_on_transitions:
            self.memory.add(transition)
        if not self.update_on_steps:
            return {}
        return self._update(time_step)

    def update_episode(self, episode: Episode, episode_num: int, time_step: int):
        if self.memory.update_on_episodes:
            self.memory.add(episode)
        if not self.update_on_episodes:
            return {}
        return self._update(time_step)

    def to(self, device: torch.device):
        self.qnetwork.to(device)
        self.qtarget.to(device)
        self.mixer.to(device)
        self.target_mixer.to(device)
        if self.ir_module is not None:
            self.ir_module.to(device)
        self.device = device
        return self

    def randomize(self):
        self.qnetwork.randomize()
        self.qtarget.randomize()
        self.mixer.randomize()
        self.target_mixer.randomize()
        if self.ir_module is not None:
            self.ir_module.randomize()
