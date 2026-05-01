from copy import deepcopy
from dataclasses import KW_ONLY, dataclass, field
from typing import Literal
import logging

import numpy as np
import torch
from marlenv import Episode, Observation, State, Transition

from marl.agents import DQNAgent
from marl.models import Batch, IRModule, Mixer, Policy, QNetwork, ReplayMemory, Trainer
from marl.optimism import VBE

from .qtarget_updater import SoftUpdate, TargetParametersUpdater


@dataclass
class DQN[M: ReplayMemory](Trainer[np.int64]):
    qnetwork: QNetwork
    train_policy: Policy
    memory: M
    _: KW_ONLY
    optimiser_type: Literal["adam", "rmsprop"] = "adam"
    gamma: float = 0.99
    batch_size: int = 64
    lr: float = 1e-4
    train_interval: tuple[int, Literal["step", "episode", "both"]] = (5, "step")
    target_updater: TargetParametersUpdater = field(default_factory=lambda: SoftUpdate(1e-2))
    double_qlearning: bool = True
    mixer: Mixer | None = None
    ir_module: IRModule | None = None
    grad_norm_clipping: float | None = None
    vbe: VBE | None = None
    test_policy: Policy | None = None

    def __post_init__(self):
        match self.train_interval:
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
        self.qtarget = deepcopy(self.qnetwork)
        self.policy = self.train_policy
        if self.test_policy is None:
            self.test_policy = self.train_policy
        self.target_mixer = deepcopy(self.mixer)
        self.update_on_steps = self.train_interval[1] == "step"
        self.update_on_episodes = self.train_interval[1] == "episode"

        # Parameters and optimiser
        self.target_updater.add_parameters(self.qnetwork.parameters(), self.qtarget.parameters())
        if self.mixer is not None:
            assert self.target_mixer is not None
            self.target_updater.add_parameters(self.mixer.parameters(), self.target_mixer.parameters())
        match self.optimiser_type:
            case "adam":
                self.optimiser = torch.optim.Adam(self.target_updater.parameters, lr=self.lr)
            case "rmsprop":
                self.optimiser = torch.optim.RMSprop(self.target_updater.parameters, lr=self.lr, eps=1e-5)
            case other:
                raise ValueError(f"Unknown optimiser: {other}. Expected 'adam' or 'rmsprop'.")
        if self.mixer is not None:
            self.name = self.mixer.name
        if self.ir_module is not None:
            self.name = f"{self.name}-{self.ir_module.name}"

    def _update(self, time_step: int) -> dict[str, float]:
        if not self.memory.can_sample(self.batch_size):
            return {}
        batch = self.memory.sample(self.batch_size).to(self.qnetwork.device)
        batch, logs = self._prepare_batch(batch)
        logs = logs | self.train(time_step, batch)
        if self.ir_module is not None:
            logs = logs | self.ir_module.update(batch, time_step)
        logs = logs | self.policy.update(time_step)
        logs = logs | self.target_updater.update(time_step)
        return logs

    def _compute_qtargets(self, batch: Batch):
        # We use the all_obs_ and all_extras_ to handle the case of recurrent qnetworks that require the first element of the sequence.
        next_qvalues = self.qtarget.batch_forward(batch.all_obs, batch.all_extras, masks=batch.all_masks)[1:]
        # For double q-learning, we use the qnetwork to select the best action. Otherwise, we use the target qnetwork.
        if self.double_qlearning:
            # It is necessary to switch to eval mode for some layers such as NoisyLayers.
            # Not switching to eval mode will cause the predicted Q-values to be off and
            # will cause torch to crash with a RuntimeError because of version mismatch.
            self.qnetwork.eval()
            qvalues_for_index = self.qnetwork.batch_forward(batch.all_obs, batch.all_extras, masks=batch.all_masks)[1:]
            self.qnetwork.train()
        else:
            qvalues_for_index = next_qvalues
        qvalues_for_index[~batch.next_available_actions] = -torch.inf
        indices = torch.argmax(qvalues_for_index, dim=-1, keepdim=True)
        next_values = torch.gather(next_qvalues, -1, indices).squeeze(-1)
        if self.target_mixer is not None:
            next_values = self.target_mixer.forward(
                next_values,
                batch.next_states,
                batch.next_states_extras,
                **self.get_mixing_kwargs(batch, next_qvalues, is_next=True),
            )
        assert batch.rewards.shape == next_values.shape == batch.dones.shape == batch.masks.shape
        return batch.rewards + self.gamma * next_values * batch.not_dones

    def _prepare_batch(self, batch: Batch):
        logs = dict[str, float]()
        if self.mixer is None:
            batch = batch.for_individual_learners()
        if self.ir_module is not None:
            ir = self.ir_module.compute(batch)
            logs.update({"ir_mean": ir.mean().item(), "ir_min": ir.min().item(), "ir_max": ir.max().item()})
            while ir.dim() < batch.rewards.dim():  # Adjust the dimensions
                ir = ir.unsqueeze(-1)
            batch.rewards = batch.rewards + ir
        return batch, logs

    def get_mixing_kwargs(self, batch: Batch, all_qvalues: torch.Tensor, is_next: bool = False):
        return {}

    def _compute_qvalues(self, batch: Batch):
        all_qvalues = self.qnetwork.batch_forward(batch.obs, batch.extras, masks=batch.masks)
        qvalues = torch.gather(all_qvalues, dim=-1, index=batch.actions.unsqueeze(-1)).squeeze(-1)
        if self.mixer is not None:
            qvalues = self.mixer.forward(qvalues, batch.states, batch.states_extras, **self.get_mixing_kwargs(batch, all_qvalues))
        return all_qvalues, qvalues

    def _compute_td_loss(self, qvalues: torch.Tensor, qtargets: torch.Tensor, batch: Batch):
        assert qtargets.grad_fn is None, "qtargets should not have a gradient function !"
        # Compute the loss
        td_error = qvalues - qtargets
        td_error = td_error * batch.masks
        squared_error = td_error**2
        if batch.importance_sampling_weights is not None:
            assert squared_error.shape == batch.importance_sampling_weights.shape
            squared_error = squared_error * batch.importance_sampling_weights
        loss = squared_error.sum() / batch.masks_sum
        return loss, td_error

    def train(self, time_step: int, batch: Batch):
        _, qvalues = self._compute_qvalues(batch)
        with torch.no_grad():
            qtargets = self._compute_qtargets(batch)
        td_loss, td_error = self._compute_td_loss(qvalues, qtargets, batch)
        logs = {"td-loss": float(td_loss.item())}
        self.optimiser.zero_grad()
        td_loss.backward()
        if self.grad_norm_clipping is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.target_updater.parameters, self.grad_norm_clipping)
            logs["grad_norm"] = grad_norm.item()
        self.optimiser.step()
        logs = logs | self.memory.update(time_step, td_error=td_error)
        if self.vbe is not None:
            logs = logs | self.vbe.update(batch)
        return logs

    def update_step(self, transition: Transition, time_step: int) -> dict[str, float]:
        logs = dict[str, float]()
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

    def make_agent(self):
        return DQNAgent(
            qnetwork=self.qnetwork,
            train_policy=self.policy,
            test_policy=self.test_policy,
            vbe=self.vbe,
        )

    def value(self, obs: Observation, state: State) -> float:
        try:
            data, extras = obs.as_tensors(self.device)
            state_data, state_extras = state.as_tensors(self.device)
            with torch.no_grad():
                qvalues = self.qnetwork.forward(data.unsqueeze(0), extras.unsqueeze(0))
                max_qvalues = qvalues.max(dim=-1).values
                if self.mixer is None:
                    return float(max_qvalues.mean().item())
                value = self.mixer.forward(
                    max_qvalues, state_data, state_extras, all_qvalues=qvalues, one_hot_actions=torch.zeros_like(qvalues)
                )
                return float(value.item())
        except Exception:
            logging.warning("Error while computing value, returning 0.0 instead")
            return 0.0
