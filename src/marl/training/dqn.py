from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Optional

import torch
from marlenv import Episode, Transition

from marl.agents import DQNAgent, RDQNAgent, Agent
from marl.models import Mixer, Policy, PrioritizedMemory, QNetwork, ReplayMemory, RecurrentQNetwork, IRModule
from marl.models.batch import Batch
from marl.models.trainer import Trainer

from .qtarget_updater import SoftUpdate, TargetParametersUpdater


@dataclass
class DQN[B: Batch](Trainer):
    qnetwork: QNetwork
    policy: Policy
    memory: ReplayMemory[Any, B]
    gamma: float
    batch_size: int
    target_updater: TargetParametersUpdater
    double_qlearning: bool
    mixer: Optional[Mixer]
    ir_module: Optional[IRModule]
    grad_norm_clipping: Optional[float]

    def __init__(
        self,
        qnetwork: QNetwork,
        train_policy: Policy,
        memory: ReplayMemory[Any, B],
        mixer: Optional[Mixer] = None,
        gamma: float = 0.99,
        batch_size: int = 64,
        lr: float = 1e-4,
        optimiser: Literal["adam", "rmsprop"] = "adam",
        target_updater: Optional[TargetParametersUpdater] = None,
        double_qlearning: bool = False,
        train_interval: tuple[int, Literal["step", "episode", "both"]] = (5, "step"),
        ir_module: Optional[IRModule] = None,
        grad_norm_clipping: Optional[float] = None,
    ):
        super().__init__()
        match train_interval:
            case (n, "step"):
                self.step_update_interval = n
                self.episode_update_interval = 0
            case (n, "episode"):
                self.step_update_interval = 0
                self.episode_update_interval = n
            case (n, "both"):
                self.step_update_interval = n
                self.episode_update_interval = n
            case other:
                raise ValueError(f"Unknown train_interval: {other}. Expected (int, 'step' | 'episode').")
        self.qnetwork = qnetwork
        self.qtarget = deepcopy(qnetwork)
        self._device = qnetwork.device
        self.policy = train_policy
        self.memory = memory
        self.gamma = gamma
        self.batch_size = batch_size
        if target_updater is None:
            target_updater = SoftUpdate(1e-2)
        self.target_updater = target_updater
        self.double_qlearning = double_qlearning
        self.mixer = mixer
        self.target_mixer = deepcopy(mixer)
        self.ir_module = ir_module
        self.update_on_steps = train_interval[1] == "step"
        self.update_on_episodes = train_interval[1] == "episode"

        # Parameters and optimiser
        self.grad_norm_clipping = grad_norm_clipping
        self.target_updater.add_parameters(qnetwork.parameters(), self.qtarget.parameters())
        if self.mixer is not None:
            assert self.target_mixer is not None
            self.target_updater.add_parameters(self.mixer.parameters(), self.target_mixer.parameters())
        match optimiser:
            case "adam":
                self.optimiser = torch.optim.Adam(self.target_updater.parameters, lr=lr, weight_decay=1e-4)
            case "rmsprop":
                self.optimiser = torch.optim.RMSprop(self.target_updater.parameters, lr=lr, eps=1e-5)
            case other:
                raise ValueError(f"Unknown optimiser: {other}. Expected 'adam' or 'rmsprop'.")

    def _update(self, time_step: int):
        if not self.memory.can_sample(self.batch_size):
            return {}
        batch = self.memory.sample(self.batch_size).to(self.qnetwork._device)
        logs = self.train(batch)
        if self.ir_module is not None:
            logs = logs | self.ir_module.update(batch, time_step)
        logs = logs | self.policy.update(time_step)
        logs = logs | self.target_updater.update(time_step)
        return logs

    def next_values(self, batch: Batch):
        # We use the all_obs_ and all_extras_ to handle the case of recurrent qnetworks that require the first element of the sequence.
        next_qvalues = self.qtarget.batch_forward(batch.all_obs, batch.all_extras)[1:]
        # For double q-learning, we use the qnetwork to select the best action. Otherwise, we use the target qnetwork.
        if self.double_qlearning:
            qvalues_for_index = self.qnetwork.batch_forward(batch.all_obs, batch.all_extras)[1:]
        else:
            qvalues_for_index = next_qvalues
        qvalues_for_index[batch.next_available_actions == 0.0] = -torch.inf
        indices = torch.argmax(qvalues_for_index, dim=-1, keepdim=True)
        next_values = torch.gather(next_qvalues, -1, indices).squeeze(-1)
        if self.target_mixer is not None:
            next_values = self.target_mixer.forward(
                next_values,
                batch.next_states,
                one_hot_actions=batch.one_hot_actions,
                next_qvalues=next_qvalues,
            )
        return next_values

    def train(self, batch: Batch) -> dict[str, Any]:
        if self.ir_module is not None:
            ir = self.ir_module.compute(batch)
            # If there is a single objective, then squeeze it
            ir = ir.squeeze()
            batch.rewards = batch.rewards + ir
        if self.mixer is None:
            # Call this after the IR module for shape reasons
            batch = batch.for_individual_learners()
        # Qvalues and qvalues with target network computation
        qvalues = self.qnetwork.batch_forward(batch.obs, batch.extras)
        qvalues = torch.gather(qvalues, dim=-1, index=batch.actions.unsqueeze(-1)).squeeze(-1)
        if self.mixer is not None:
            qvalues = self.mixer.forward(qvalues, batch.states, one_hot_actions=batch.one_hot_actions, next_qvalues=qvalues)

        # Qtargets computation
        next_values = self.next_values(batch)
        assert batch.rewards.shape == next_values.shape == batch.dones.shape == qvalues.shape == batch.masks.shape
        qtargets = batch.rewards + self.gamma * next_values * batch.not_dones
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
            grad_norm = torch.nn.utils.clip_grad_norm_(self.target_updater.parameters, self.grad_norm_clipping)
            logs["grad_norm"] = grad_norm.item()
        self.optimiser.step()
        if isinstance(self.memory, PrioritizedMemory):
            logs = logs | self.memory.update(td_error)
        return logs

    def update_step(self, transition: Transition, time_step: int) -> dict[str, Any]:
        logs = {}
        if self.ir_module is not None:
            logs = logs | self.ir_module.update_step(transition, time_step)
        if self.memory.update_on_transitions:
            self.memory.add(transition)
        if self.update_on_steps and time_step % self.step_update_interval == 0:
            logs = logs | self._update(time_step)
        return logs

    def update_episode(self, episode: Episode, episode_num: int, time_step: int):
        logs = {}
        if self.ir_module is not None:
            logs = logs | self.ir_module.update_episode(episode, time_step)
        if self.memory.update_on_episodes:
            self.memory.add(episode)
        if self.update_on_episodes and episode_num % self.episode_update_interval == 0:
            logs = logs | self._update(time_step)
        return logs

    def make_agent(self, test_policy: Optional[Policy] = None) -> Agent:
        if isinstance(self.qnetwork, RecurrentQNetwork):
            return RDQNAgent(
                qnetwork=self.qnetwork,
                train_policy=self.policy,
                test_policy=test_policy,
            )
        return DQNAgent(
            qnetwork=self.qnetwork,
            train_policy=self.policy,
            test_policy=test_policy,
        )

    def to(self, device: torch.device):
        if self.ir_module is not None:
            self.ir_module.to(device)
        return super().to(device)

    def randomize(self, method: Literal["xavier", "orthogonal"] = "xavier"):
        super().randomize(method)
        if self.ir_module is not None:
            self.ir_module.randomize(method)
