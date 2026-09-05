"""Networks for LAN (Avalos et al., TMLR 2023, Appendix B)."""

from dataclasses import dataclass, field
from math import prod

import torch

from marl.models.nn import NN, RecurrentQNetwork


@dataclass(eq=False)
class LocalAdvantageNetwork(RecurrentQNetwork):
    """Shared local FC/GRU policy. Extras should contain agent ID and previous action."""

    duelling: bool = False
    hidden_size: int = 64
    mean_center: bool = False

    def __post_init__(self):
        """Construct the local network without a local value head. @ai-generated"""
        if self.duelling or self.noisy or self.independent or self.n_objectives != 1:
            raise ValueError("LAN supports shared, non-noisy, scalar-reward advantages without a duelling head.")
        super().__post_init__()
        self.fc = torch.nn.Linear(self.obs_size + self.extras_size, self.hidden_size)
        self.gru = torch.nn.GRU(self.hidden_size, self.hidden_size)
        self.head = torch.nn.Linear(self.hidden_size, self.n_actions)

    def features(self, obs: torch.Tensor, extras: torch.Tensor, hidden: torch.Tensor | None = None):
        """Unroll time-major local histories; no cross-agent communication. @ai-generated"""
        prefix = obs.shape[: -len(self.obs_shape)]
        inputs = torch.cat((obs.reshape(*prefix, self.obs_size), extras.reshape(*prefix, self.extras_size)), -1)
        encoded = torch.relu(self.fc(inputs))
        output, hidden = self.gru(encoded.reshape(encoded.shape[0], -1, self.hidden_size), hidden)
        histories = output.reshape(*prefix, self.hidden_size)
        advantages = self.head(histories)
        if self.mean_center:
            advantages = advantages - advantages.mean(-1, keepdim=True)
        return advantages, histories, hidden

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, /, **kwargs):
        """Advance the acting agent's recurrent state. @ai-generated"""
        advantages, _, self._hidden_states = self.features(obs, extras, self._hidden_states)
        return advantages

    def to(self, device: torch.device):
        """Move both parameters and live/saved rollout histories. @ai-generated"""
        super().to(device)
        if self._hidden_states is not None:
            self._hidden_states = self._hidden_states.to(device)
        if self._saved_hidden_states is not None:
            self._saved_hidden_states = self._saved_hidden_states.to(device)
        return self

    def __hash__(self):
        return id(self)


@dataclass(eq=False)
class LANValue(NN):
    """Shared agent embedding, sum pooling, then a centralized state/history value."""

    output_shape: tuple[int, ...] = field(init=False, default=(1,))
    obs_shape: tuple[int, ...]
    extras_shape: tuple[int, ...]
    state_shape: tuple[int, ...]
    state_extras_shape: tuple[int, ...] = (0,)
    hidden_size: int = 64
    embedding_size: int = 128

    def __post_init__(self):
        """Build Appendix B's shared embedding and two-layer value MLP. @ai-generated"""
        super().__post_init__()
        self.embedding = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size + prod(self.obs_shape) + prod(self.extras_shape), self.embedding_size),
            torch.nn.ReLU(),
        )
        self.value = torch.nn.Sequential(
            torch.nn.Linear(
                self.embedding_size + prod(self.state_shape) + prod(self.state_extras_shape), self.embedding_size
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, 1),
        )

    def forward(self, histories, obs, extras, states, state_extras):
        """Compute V(s, tau), retaining gradients into local history encoders. @ai-generated"""
        prefix = histories.shape[:-1]
        inputs = torch.cat((histories, obs.reshape(*prefix, -1), extras.reshape(*prefix, prod(self.extras_shape))), -1)
        pooled = self.embedding(inputs).sum(dim=-2)
        joint_prefix = pooled.shape[:-1]
        return self.value(
            torch.cat(
                (
                    pooled,
                    states.reshape(*joint_prefix, prod(self.state_shape)),
                    state_extras.reshape(*joint_prefix, prod(self.state_extras_shape)),
                ),
                -1,
            )
        )

    def __hash__(self):
        return id(self)
