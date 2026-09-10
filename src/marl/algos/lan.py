"""Local Advantage Networks, https://arxiv.org/abs/2112.12458v3."""

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import torch
from marlenv import Episode, Observation, State, Transition

from marl import policy
from marl.env import EnvConfig
from marl.models import Batch, Policy, Trainer
from marl.nn.lan import LANValue, LocalAdvantageNetwork

from .dqn import DQN
from .qtarget_updater import HardUpdate, TargetParametersUpdater


@dataclass(kw_only=True)
class LAN(DQN[None]):
    """Jointly learn V(s, tau) + A_a(tau_a, u_a) using one-step double DQN."""

    qnetwork: LocalAdvantageNetwork = field(kw_only=False)  # pyright: ignore[reportIncompatibleVariableOverride]
    value_network: LANValue = field(kw_only=True)
    lr: float = 5e-4
    batch_size: int = 32
    memory_size: int | Literal["auto"] = 5000
    train_interval: tuple[int, Literal["step", "episode"]] = (1, "episode")
    grad_norm_clipping: float | None = 10.0
    train_policy: Policy = field(default_factory=lambda: policy.EpsilonGreedy.linear(1.0, 0.05, 50_000))
    target_updater: TargetParametersUpdater = field(default_factory=lambda: HardUpdate(200), hash=False)
    updates_per_episode: int = 2

    def __post_init__(self):
        """Register both online/target components with the optimizer and target updater. @ai-generated"""
        if self.mixer is not None or tuple(self.train_interval) != (1, "episode"):
            raise ValueError("LAN requires no mixer and training after each episode.")
        self.train_interval = (1, "episode")
        if self.updates_per_episode < 1:
            raise ValueError("updates_per_episode must be positive")
        if self.value_network.hidden_size != self.qnetwork.hidden_size:
            raise ValueError("The advantage and value networks must agree on hidden_size")
        super().__post_init__()
        self.target_value = deepcopy(self.value_network)
        self.target_updater.add_parameters(self.value_network.parameters(), self.target_value.parameters())
        self.optimiser = self._make_optimiser()
        self._initial_targets_synced = False

    @classmethod
    def from_env(
        cls, env: EnvConfig, *, hidden_size: int = 64, embedding_size: int = 128, mean_center: bool = False, **kwargs
    ):
        """Build serializable LAN networks for a scalar-reward environment. @ai-generated"""
        if env.n_objectives != 1:
            raise ValueError("LAN requires a scalar shared reward")
        return cls(
            LocalAdvantageNetwork.from_env(env, hidden_size=hidden_size, mean_center=mean_center),
            value_network=LANValue(
                env.observation_shape,
                env.extras_shape,
                env.state_shape,
                env.state_extra_shape,
                hidden_size,
                embedding_size,
            ),
            **kwargs,
        )

    @property
    def name(self):
        return "LAN-mean" if self.qnetwork.mean_center else "LAN"

    def compile(self, fullgraph: bool = True):
        """Use eager execution: PyTorch's native GRU cannot be captured with fullgraph. @ai-generated"""
        return

    def save(self, directory: Path):
        """Keep online and target weights separate despite identical network classes. @ai-generated"""
        directory.mkdir(parents=True, exist_ok=True)
        target_directory = directory / "lan-targets"
        target_directory.mkdir(exist_ok=True)
        self.qnetwork.save(directory)
        self.value_network.save(directory)
        self.qtarget.save(target_directory)
        self.target_value.save(target_directory)

    def load(self, directory: Path):
        """Restore online and target weights from their separate checkpoint locations. @ai-generated"""
        self.qnetwork.load(directory)
        self.value_network.load(directory)
        self.qtarget.load(directory / "lan-targets")
        self.target_value.load(directory / "lan-targets")
        self._initial_targets_synced = True

    def to(self, device: torch.device):
        """Move every network and retain Adam/RMSprop state across device transfers. @ai-generated"""
        device = torch.device(device)
        optimizer_state = self.optimiser.state_dict()
        Trainer.to(self, device)
        self.target_updater._parameters = []
        self.target_updater._target_params = []
        self.target_updater.add_parameters(self.qnetwork.parameters(), self.qtarget.parameters())
        self.target_updater.add_parameters(self.value_network.parameters(), self.target_value.parameters())
        self.optimiser = self._make_optimiser()
        self.optimiser.load_state_dict(optimizer_state)
        return self

    def _compute_qvalues(self, batch: Batch):
        """Equation 1: individual proxies share a single centralized value. @ai-generated"""
        advantages, histories, _ = self.qnetwork.features(batch.obs, batch.extras)
        value = self.value_network(histories, batch.obs, batch.extras, batch.states, batch.states_extras)
        chosen = advantages.gather(-1, batch.actions.unsqueeze(-1)).squeeze(-1)
        return advantages, chosen + value

    @torch.no_grad()
    def _compute_qtargets(self, batch: Batch):
        """Equation 3, unrolling from reset, masking illegal actions and true terminals. @ai-generated"""
        target_network = cast(LocalAdvantageNetwork, self.qtarget)
        target_advantages, histories, _ = target_network.features(batch.all_obs, batch.all_extras)
        target_advantages = target_advantages[1:]
        if self.double_qlearning:
            selection, _, _ = self.qnetwork.features(batch.all_obs, batch.all_extras)
            selection = selection[1:]
        else:
            selection = target_advantages
        indices = selection.masked_fill(~batch.next_available_actions, -torch.inf).argmax(-1, keepdim=True)
        next_advantages = target_advantages.gather(-1, indices).squeeze(-1)
        next_value = self.target_value(
            histories[1:], batch.next_obs, batch.next_extras, batch.next_states, batch.next_states_extras
        )
        return batch.rewards + self.gamma * (next_value + next_advantages) * batch.not_dones

    def _update(self, time_step: int):
        """Synchronize initial targets after the runner has seeded/randomized the agent. @ai-generated"""
        if not self._initial_targets_synced:
            self.qtarget.load_state_dict(self.qnetwork.state_dict())
            self.target_value.load_state_dict(self.value_network.state_dict())
            self._initial_targets_synced = True
        return super()._update(time_step)

    def update_episode(self, episode: Episode, episode_num: int, time_step: int):
        """Sample a fresh batch for each of Appendix B's two updates. @ai-generated"""
        self.memory.add_episode(episode)
        logs = {}
        for _ in range(self.updates_per_episode):
            logs.update(self._update(time_step))
        return logs

    def update_step(self, transition: Transition, time_step: int):
        """Anneal exploration in environment steps even before replay warmup finishes. @ai-generated"""
        return self.policy.update(time_step)

    def value(self, obs: Observation, state: State) -> float:
        """Report centralized value without advancing the acting recurrent history. @ai-generated"""
        with torch.no_grad():
            data, extras = obs.as_tensors(self.device)
            states, state_extras = state.as_tensors(self.device)
            data, extras = data.unsqueeze(0), extras.unsqueeze(0)
            _, histories, _ = self.qnetwork.features(data, extras, self.qnetwork._hidden_states)
            return self.value_network(histories, data, extras, states.unsqueeze(0), state_extras.unsqueeze(0)).item()
