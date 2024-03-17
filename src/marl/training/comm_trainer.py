from typing import Any, Literal, Optional
import torch
from rlenv import Transition, Episode
from marl.models import QNetwork, ReplayMemory, Policy, Trainer, Mixer, CommNetwork, PrioritizedMemory
from marl.intrinsic_reward import IRModule
from marl.models.batch import EpisodeBatch, Batch
from marl.utils import defaults_to
from .qtarget_updater import TargetParametersUpdater, SoftUpdate
from dataclasses import dataclass
from serde import serialize
import numpy as np
from copy import deepcopy


@serialize
@dataclass
class CommTrainer(Trainer):
    qnetwork: QNetwork
    comm_network: CommNetwork
    memory: ReplayMemory
    gamma: float
    batch_size: int
    target_updater: Optional[Any]
    double_qlearning: bool
    mixer: Optional[Mixer]
    ir_module: Optional[IRModule]
    grad_norm_clipping: Optional[float]

    def __init__(
        self,
        qnetwork: QNetwork,
        comm_network: CommNetwork,
        train_policy: Policy,
        memory: ReplayMemory,
        gamma: float = 0.99,
        batch_size: int = 32,
        lr: float = 1e-4,
        optimiser: Literal["adam", "rmsprop"] = "adam",
        target_updater: Optional[TargetParametersUpdater] = None,
        double_qlearning: bool = False,
        mixer: Optional[Mixer] = None,
        ir_module: Optional[IRModule] = None,
        grad_norm_clipping: Optional[float] = None,
        update_interval: int = 5,
        update_type: Literal["step", "episode"] = "step",
    ):
        super().__init__(update_type, update_interval)
        self.qnetwork = qnetwork
        self.qtarget = deepcopy(qnetwork)
        self.comm_network = comm_network
        self.comm_target = deepcopy(comm_network)

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

        self.grad_norm_clipping = grad_norm_clipping
        self.target_updater.add_parameters(qnetwork.parameters(), self.qtarget.parameters())
        if mixer is not None and self.target_mixer is not None:
            self.target_updater.add_parameters(mixer.parameters(), self.target_mixer.parameters())
        match optimiser:
            case "adam":
                self.optimiser = torch.optim.Adam(self.target_updater.parameters, lr=lr)
            case "rmsprop":
                self.optimiser = torch.optim.RMSprop(self.target_updater.parameters, lr=lr)
            case other:
                raise ValueError(f"Unknown optimiser: {other}. Expected 'adam' or 'rmsprop'.")
        

    def _update(self, time_step: int) -> dict[str, Any]:
        logs, td_error = self.optimise_qnetwork()
        logs = logs | self.policy.update(time_step)
        logs = logs | self.target_updater.update(time_step)
        if isinstance(self.memory, PrioritizedMemory):
            self.memory.update(td_error)
            logs["per-alpha"] = self.memory.alpha.value
            logs["per-beta"] = self.memory.beta.value
        return logs

    def _next_state_value(self, batch: Batch) -> torch.Tensor:
        messages = self.comm_target.encodeTensor(batch.obs_, batch.extras_)
        batch.extras_ = self.add_message_to_extra(batch.extras_, messages)
        next_qvalues = self.qtarget.batch_forward(batch.obs_, batch.extras_)
        # For double q-learning, we use the qnetwork to select the best action. Otherwise, we use the target qnetwork.
        if self.double_qlearning:
            qvalues_for_index = self.qnetwork.batch_forward(batch.obs_, batch.extras_)
        else:
            qvalues_for_index = next_qvalues
        # For episode batches, the batch includes the initial observation to compute the hidden state at t=0.
        # We remove it when considering the next qvalues.
        if isinstance(batch, EpisodeBatch):
            next_qvalues = next_qvalues[1:]
            qvalues_for_index = qvalues_for_index[1:]
        qvalues_for_index[batch.available_actions_ == 0.0] = -torch.inf
        indices = torch.argmax(qvalues_for_index, dim=-1, keepdim=True)
        next_values = torch.gather(next_qvalues, -1, indices).squeeze(-1)
        if self.target_mixer is not None:
            next_values = self.target_mixer.forward(next_values, batch.states_, batch.one_hot_actions, next_qvalues)
        return next_values
    
    @staticmethod
    def add_message_to_extra(extras: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        message_tensor = message.detach().cpu()
        if extras.shape[1] > 0:
            extras = torch.cat([extras, message_tensor], dim=-1)
        else:
            extras = message_tensor
        return extras

    def optimise_qnetwork(self) -> tuple[dict[str, Any], torch.Tensor]:
        batch = self.memory.sample(self.batch_size)
        if self.ir_module is not None:
            batch.rewards = batch.rewards + self.ir_module.compute(batch)

        # Compute message from current observation using CommNetwork
        message = self.comm_network.encodeTensor(batch.obs, batch.extras)

        # Add message to the observation's extras
        batch.extras = self.add_message_to_extra(batch.extras, message)

        # Qvalues computation
        all_qvalues = self.qnetwork.batch_forward(batch.obs, batch.extras)
        batch.actions = batch.actions.to(all_qvalues.device)
        qvalues = torch.gather(all_qvalues, dim=-1, index=batch.actions).squeeze(-1)
        if self.mixer is not None:
            qvalues = self.mixer.forward(qvalues, batch.states, batch.one_hot_actions, all_qvalues)

        # Qtargets computation
        next_values = self._next_state_value(batch)
        batch.rewards = batch.rewards.to(next_values.device)
        batch.dones = batch.dones.to(next_values.device)
        qtargets = batch.rewards + self.gamma * next_values * (1 - batch.dones)
        qtargets = qtargets.to(qvalues.device)

        # Compute the loss
        td_error = qvalues - qtargets.detach()
        batch.masks = batch.masks.to(td_error.device)   
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
        if len(self.memory) < self.batch_size:
            return {}
        self.update_num += 1
        if self.update_num % self.update_interval != 0:
            return {}
        return self._update(time_step)

    def update_episode(self, episode: Episode, episode_num: int, time_step: int) -> dict[str, Any]:
        if not self.update_on_episodes:
            return {}
        self.memory.add(episode)
        return {}

    def to(self, device: torch.device):
        if self.mixer is not None:
            self.mixer.to(device)
        # if self.target_mixer is not None:
        #     self.target_mixer.to(device)
        self.qnetwork.to(device)
        self.comm_network.to(device)
        # self.qtarget.to(device)
        return self

    def randomize(self):
        self.qnetwork.randomize()
        self.comm_network.randomize()
        # self.qtarget.randomize()
        if self.mixer is not None:
            self.mixer.randomize()
        # if self.target_mixer is not None:
        #     self.target_mixer.randomize()