from copy import copy, deepcopy
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from typing import Literal

import torch
from marlenv import Episode, Observation, State, Transition

from marl import policy
from marl.models import Agent, Batch, Mixer, Policy, QNetwork, ReplayMemory, Trainer
from marl.models.batch import EpisodeBatch
from marl.utils.tuning import tuning

from .optimism import VBE
from .qtarget_updater import SoftUpdate, TargetParametersUpdater


@dataclass(unsafe_hash=True)
class DQN[M: (Mixer | None)](Trainer):
    qnetwork: QNetwork
    memory: ReplayMemory
    _: KW_ONLY
    mixer: M = None  # type: ignore
    train_policy: Policy = field(default_factory=lambda: policy.EpsilonGreedy.constant(0.1))
    lr: float = field(default=1e-4, metadata=tuning(1e-5, 1e-2, log=True))
    batch_size: int = field(default=64, metadata=tuning(16, 256))
    double_qlearning: bool = True
    test_policy: Policy = field(default_factory=policy.ArgMax)
    target_updater: TargetParametersUpdater = field(default_factory=lambda: SoftUpdate(1e-2), hash=False)
    optimiser_type: Literal["adam", "rmsprop"] = "adam"
    vbe: VBE | None = None

    def __post_init__(self):
        super().__post_init__()
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
        self.optimiser = self._make_optimiser()

    def save(self, directory: Path):
        """Store targets separately so they cannot overwrite online network files. @ai-generated"""
        super().save(directory)
        target_directory = directory / "dqn-targets"
        target_directory.mkdir(exist_ok=True)
        self.qtarget.save(target_directory)
        if self.target_mixer is not None:
            self.target_mixer.save(target_directory)
        self.qnetwork.save(directory)
        if self.mixer is not None:
            self.mixer.save(directory)

    def load(self, directory: Path):
        """Load separate targets, or synchronize them when reading legacy checkpoints. @ai-generated"""
        super().load(directory)
        target_directory = directory / "dqn-targets"
        if target_directory.exists():
            self.qtarget.load(target_directory)
            if self.target_mixer is not None:
                self.target_mixer.load(target_directory)
        else:
            self.qtarget.load_state_dict(self.qnetwork.state_dict())
            if self.mixer is not None and self.target_mixer is not None:
                self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _make_optimiser(self, fused: bool = False):
        match self.optimiser_type:
            case "adam":
                return torch.optim.Adam(self.target_updater.parameters, lr=self.lr, fused=fused)
            case "rmsprop":
                return torch.optim.RMSprop(self.target_updater.parameters, lr=self.lr, eps=1e-5)
        raise ValueError(f"Unknown optimiser: {self.optimiser_type}. Expected 'adam' or 'rmsprop'.")

    @property
    def name(self):
        name = "DQN" if self.mixer is None else self.mixer.name + f"-{self.qnetwork.name}"
        if self.double_qlearning:
            name += "-double"
        if self.qnetwork.duelling:
            name += "-duelling"
        if self.qnetwork.noisy:
            name += "-noisy"
        if self.ir_module is not None:
            name += f"-{self.ir_module.name}"
        if self.vbe is not None:
            name += f"-{self.vbe.name}"
        return name

    @property
    def n_actions(self):
        return self.qnetwork.n_actions

    def _update(self, time_step: int) -> dict[str, float]:
        """Train with shaped rewards, but update the IR module with extrinsic rewards. @ai-edited"""
        if not self.memory.can_sample(self.batch_size):
            return {}
        batch = self.memory.sample(self.batch_size).to(self.device)
        if self.mixer is None:
            batch = batch.for_individual_learners()
        extrinsic_rewards = batch.rewards.clone() if self.ir_module is not None else None
        batch, logs = self._prepare_batch(batch)
        logs = logs | self.train(time_step, batch)
        if self.ir_module is not None:
            assert extrinsic_rewards is not None
            ir_batch = copy(batch)
            ir_batch.rewards = extrinsic_rewards
            logs = logs | self.ir_module.update(ir_batch, time_step)
        if self.vbe is not None:
            logs = logs | self.vbe.update(batch, qnetwork=self.qnetwork)
        logs = logs | self.policy.update(time_step)
        logs = logs | self.target_updater.update(time_step)
        return logs

    def _compute_qtargets(self, batch: Batch):
        """Bootstrap legal actions and pass the selected joint action to mixers. @ai-generated"""
        # We use the all_obs_ and all_extras_ to handle the case of recurrent qnetworks that require the first element of the sequence.
        next_qvalues = self.qtarget.batch_qvalues(batch.all_obs, batch.all_extras, masks=batch.all_masks)[1:]
        # For double q-learning, we use the qnetwork to select the best action. Otherwise, we use the target qnetwork.
        if self.double_qlearning:
            # It is necessary to switch to eval mode for some layers such as NoisyLayers.
            # Not switching to eval mode will cause the predicted Q-values to be off and
            # will cause torch to crash with a RuntimeError because of version mismatch.
            self.qnetwork.eval()
            qvalues_for_index = self.qnetwork.batch_qvalues(batch.all_obs, batch.all_extras, masks=batch.all_masks)[1:]
            self.qnetwork.train()
        else:
            qvalues_for_index = next_qvalues
        if self.qnetwork.is_multi_objective:
            qvalues_for_index = qvalues_for_index.sum(dim=-1)
        qvalues_for_index = qvalues_for_index.masked_fill(~batch.next_available_actions, -torch.inf)
        indices = torch.argmax(qvalues_for_index, dim=-1, keepdim=True)
        if self.qnetwork.is_multi_objective:
            objective_indices = indices.unsqueeze(-1).expand(*indices.shape, self.qnetwork.n_objectives)
            next_values = torch.gather(next_qvalues, -2, objective_indices).squeeze(-2)
        else:
            next_values = torch.gather(next_qvalues, -1, indices).squeeze(-1)
        if self.target_mixer is not None:
            next_values = self.target_mixer.forward_batch(next_values, batch, next_qvalues, indices.squeeze(-1), is_next=True)
        assert batch.rewards.shape == next_values.shape == batch.not_dones.shape == batch.masks.shape
        gamma = batch.gamma if batch.gamma is not None else self.gamma
        if isinstance(gamma, torch.Tensor):
            while gamma.ndim < next_values.ndim:
                gamma = gamma.unsqueeze(-1)
        return batch.rewards + gamma * next_values.masked_fill(batch.dones | batch.masked_indices, 0)

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

    def _compute_qvalues(self, batch: Batch):
        """Gather the selected action while preserving any objective dimension."""
        all_qvalues = self.qnetwork.batch_qvalues(batch.obs, batch.extras, masks=batch.masks)
        if self.qnetwork.is_multi_objective:
            indices = batch.actions.unsqueeze(-1).unsqueeze(-1).expand(*batch.actions.shape, 1, self.qnetwork.n_objectives)
            qvalues = torch.gather(all_qvalues, dim=-2, index=indices).squeeze(-2)
        else:
            qvalues = torch.gather(all_qvalues, dim=-1, index=batch.actions.unsqueeze(-1)).squeeze(-1)
        if self.mixer is not None:
            qvalues = self.mixer.forward_batch(qvalues, batch, all_qvalues, batch.actions)
        return all_qvalues, qvalues

    def _compute_td_loss(self, qvalues: torch.Tensor, qtargets: torch.Tensor, batch: Batch):
        assert qtargets.grad_fn is None, "qtargets should not have a gradient function !"
        # Compute the loss
        td_error = qvalues - qtargets
        td_error = td_error * batch.masks
        squared_error = td_error**2
        if batch.importance_sampling_weights is not None:
            weights = batch.importance_sampling_weights
            if weights.ndim == 1:
                shape = [1] * squared_error.ndim
                shape[1 if isinstance(batch, EpisodeBatch) else 0] = batch.size
                weights = weights.reshape(shape)
            squared_error = squared_error * weights
        loss = squared_error.sum() / batch.n_items
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
            logs["grad_norm"] = torch.nn.utils.clip_grad_norm_(self.target_updater.parameters, self.grad_norm_clipping).item()
        self.optimiser.step()
        logs = logs | self.memory.update(time_step, td_error=td_error)
        return logs

    def update_step(self, transition: Transition, time_step: int) -> dict[str, float]:
        self.memory.add_transition(transition)
        if self.should_update_at(time_step=time_step):
            return self._update(time_step)
        return dict[str, float]()

    def update_episode(self, episode: Episode, episode_num: int, time_step: int):
        self.memory.add_episode(episode)
        if self.should_update_at(episode_num=episode_num):
            return self._update(time_step)
        return dict[str, float]()

    def make_agent(self) -> Agent:
        from marl.agents import DQNAgent

        return DQNAgent(
            qnetwork=self.qnetwork,
            train_policy=self.policy,
            test_policy=self.test_policy,
            vbe=self.vbe,
        )

    def value(self, obs: Observation, state: State) -> float:
        """Evaluate the best available action using Q-values, not raw network heads. @ai-edited"""
        data, extras = obs.as_tensors(self.device)
        state_data, state_extras = state.as_tensors(self.device)
        with torch.no_grad():
            qvalues = self.qnetwork.batch_qvalues(data.unsqueeze(0), extras.unsqueeze(0))
            available = torch.as_tensor(obs.available_actions, device=self.device, dtype=torch.bool).unsqueeze(0)
            max_qvalues, greedy_actions = qvalues.masked_fill(~available, -torch.inf).max(dim=-1)
            if self.mixer is None:
                return float(max_qvalues.mean().item())
            value = self.mixer.forward(
                max_qvalues,
                state_data,
                state_extras,
                **self.mixer.mixing_kwargs(qvalues, greedy_actions),
            )
            return float(value.item())
