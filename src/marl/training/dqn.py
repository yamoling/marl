import logging
from copy import deepcopy
from dataclasses import KW_ONLY, dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
import torch
from marlenv import Episode, Observation, State, Transition

from marl import policy
from marl.agents import DQNAgent
from marl.models import Batch, Mixer, Policy, QNetwork, ReplayMemory, Trainer
from marl.optimism import VBE

from .qtarget_updater import SoftUpdate, TargetParametersUpdater


@dataclass(unsafe_hash=True)
class DQN[M: (Mixer | None)](Trainer[npt.NDArray[np.int64]]):
    qnetwork: QNetwork
    train_policy: Policy
    memory: ReplayMemory
    mixer: M
    _: KW_ONLY
    lr: float = 1e-4
    batch_size: int = 64
    double_qlearning: bool = True
    test_policy: Policy = field(default_factory=policy.ArgMax)
    target_updater: TargetParametersUpdater = field(default_factory=lambda: SoftUpdate(1e-2), hash=False)
    optimiser_type: Literal["adam", "rmsprop"] = "adam"
    vbe: VBE | None = None

    def __post_init__(self):
        match self.train_interval:
            case (n, "step"):
                self.step_update_interval = n
                self.episode_update_interval = 0
            case (n, "episode"):
                self.step_update_interval = 0
                self.episode_update_interval = n
            case other:
                raise ValueError(f"Unknown train_interval: {other}. Expected (int, 'step' | 'episode').")
        self.qtarget = deepcopy(self.qnetwork)
        self.policy = self.train_policy
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
        else:
            self.name = "DQN"
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
        if self.vbe is not None:
            logs = logs | self.vbe.update(batch)
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
        assert batch.rewards.shape == next_values.shape == batch.not_dones.shape == batch.masks.shape
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

    def get_mixing_kwargs(self, batch: Batch, all_qvalues: torch.Tensor, is_next: bool = False) -> dict[str, torch.Tensor]:
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
        return logs

    def update_step(self, transition: Transition, time_step: int) -> dict[str, float]:
        if self.memory.update_on_transitions:
            self.memory.add(transition)
        if self.should_update_at(time_step=time_step):
            return self._update(time_step)
        return dict[str, float]()

    def update_episode(self, episode: Episode, episode_num: int, time_step: int):
        if self.memory.update_on_episodes:
            self.memory.add(episode)
        if self.should_update_at(episode_num=episode_num):
            return self._update(time_step)
        return dict[str, float]()

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
