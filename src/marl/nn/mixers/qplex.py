from dataclasses import KW_ONLY, dataclass

import numpy as np
import torch
import torch.nn as nn
from marlenv import MARLEnv, MultiDiscreteSpace

from marl.models.nn import Mixer


class DMAQSIWeight(nn.Module):
    """State-action dependent weighting used in the QPLEX advantage stream."""

    def __init__(
        self,
        state_dim: int,
        n_agents: int,
        n_actions: int,
        num_kernel: int = 10,
        adv_hypernet_layers: int = 3,
        adv_hypernet_embed: int = 64,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.action_dim = n_agents * n_actions
        self.state_action_dim = self.state_dim + self.action_dim
        self.num_kernel = num_kernel

        self.key_extractors = nn.ModuleList()
        self.agent_extractors = nn.ModuleList()
        self.action_extractors = nn.ModuleList()
        for _ in range(num_kernel):
            self.key_extractors.append(self._make_head(self.state_dim, 1, adv_hypernet_layers, adv_hypernet_embed))
            self.agent_extractors.append(self._make_head(self.state_dim, self.n_agents, adv_hypernet_layers, adv_hypernet_embed))
            self.action_extractors.append(self._make_head(self.state_action_dim, self.n_agents, adv_hypernet_layers, adv_hypernet_embed))

    @staticmethod
    def _make_head(in_features: int, out_features: int, n_layers: int, hidden_size: int) -> nn.Module:
        if n_layers == 1:
            return nn.Linear(in_features, out_features)
        if n_layers == 2:
            return nn.Sequential(
                nn.Linear(in_features, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, out_features),
            )
        if n_layers == 3:
            return nn.Sequential(
                nn.Linear(in_features, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, out_features),
            )
        raise ValueError(f"Unsupported adv_hypernet_layers={n_layers}. Expected 1, 2 or 3.")

    def forward(self, states: torch.Tensor, one_hot_actions: torch.Tensor) -> torch.Tensor:
        states = states.reshape(-1, self.state_dim)
        one_hot_actions = one_hot_actions.reshape(-1, self.action_dim)
        state_action = torch.cat([states, one_hot_actions], dim=-1)

        head_weights = []
        for key_ext, agent_ext, action_ext in zip(self.key_extractors, self.agent_extractors, self.action_extractors):
            x_key = torch.abs(key_ext(states)).repeat(1, self.n_agents) + 1e-10
            x_agent = torch.sigmoid(agent_ext(states))
            x_action = torch.sigmoid(action_ext(state_action))
            head_weights.append(x_key * x_agent * x_action)

        # Sum kernels to get per-agent weighting term in the advantage stream.
        return torch.stack(head_weights, dim=1).sum(dim=1)


@dataclass(unsafe_hash=True)
class QPlex(Mixer):
    """
    QPLEX mixer close to the official implementation in `qplex_official`.

    This follows the duplex dueling decomposition:
    - utility stream: transformed per-agent utilities
    - advantage stream: weighted advantage wrt per-agent greedy values
    """

    state_shape: int | tuple[int, ...]
    n_actions: int
    _: KW_ONLY
    embed_dim: int = 32
    hypernet_embed: int = 64
    num_kernel: int = 10
    adv_hypernet_layers: int = 3
    adv_hypernet_embed: int = 64
    is_minus_one: bool = True
    weighted_head: bool = True
    is_stop_gradient: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.state_dim = int(np.prod(self.state_shape))

        self.hyper_w_final = nn.Sequential(
            nn.Linear(self.state_dim, self.hypernet_embed),
            nn.ReLU(),
            nn.Linear(self.hypernet_embed, self.n_agents),
        )
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.hypernet_embed),
            nn.ReLU(),
            nn.Linear(self.hypernet_embed, self.n_agents),
        )
        self.si_weight = DMAQSIWeight(
            state_dim=self.state_dim,
            n_agents=self.n_agents,
            n_actions=self.n_actions,
            num_kernel=self.num_kernel,
            adv_hypernet_layers=self.adv_hypernet_layers,
            adv_hypernet_embed=self.adv_hypernet_embed,
        )

    def _calc_v(self, agent_qs: torch.Tensor) -> torch.Tensor:
        return torch.sum(agent_qs, dim=-1)

    def _calc_adv(
        self,
        agent_qs: torch.Tensor,
        max_q_i: torch.Tensor,
        states: torch.Tensor,
        one_hot_actions: torch.Tensor,
    ) -> torch.Tensor:
        adv_q = agent_qs - max_q_i
        if self.is_stop_gradient:
            adv_q = adv_q.detach()

        adv_w_final = self.si_weight(states, one_hot_actions).view(-1, self.n_agents)
        if self.is_minus_one:
            return torch.sum(adv_q * (adv_w_final - 1.0), dim=-1)
        return torch.sum(adv_q * adv_w_final, dim=-1)

    def _mix_single(
        self,
        qvalues: torch.Tensor,
        states: torch.Tensor,
        one_hot_actions: torch.Tensor,
        all_qvalues: torch.Tensor,
        available_actions: torch.Tensor | None,
    ) -> torch.Tensor:
        states_flat = states.reshape(-1, self.state_dim)
        agent_qs = qvalues.reshape(-1, self.n_agents)

        # Compute per-agent greedy baseline max_a Q_i(s, a).
        q_for_max = all_qvalues
        if available_actions is not None:
            q_for_max = q_for_max.masked_fill(~available_actions, -torch.inf)
        max_q_i = q_for_max.max(dim=-1).values.reshape(-1, self.n_agents)

        w_final = torch.abs(self.hyper_w_final(states_flat)).view(-1, self.n_agents) + 1e-10
        v = self.V(states_flat).view(-1, self.n_agents)

        if self.weighted_head:
            agent_qs = w_final * agent_qs + v
            max_q_i = w_final * max_q_i + v

        v_tot = self._calc_v(agent_qs)
        adv_tot = self._calc_adv(agent_qs, max_q_i, states_flat, one_hot_actions)
        return v_tot + adv_tot

    def forward(
        self,
        qvalues: torch.Tensor,
        states: torch.Tensor,
        /,
        one_hot_actions: torch.Tensor | None = None,
        all_qvalues: torch.Tensor | None = None,
        available_actions: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if one_hot_actions is None:
            raise ValueError("QPlex requires `one_hot_actions` for the advantage stream.")
        if all_qvalues is None:
            raise ValueError("QPlex requires `all_qvalues` to compute per-agent max Q-values.")

        if self.n_objectives == 1:
            mixed = self._mix_single(qvalues, states, one_hot_actions, all_qvalues, available_actions)
            return mixed.view(*qvalues.shape[:-1])

        outputs = []
        base_shape = qvalues.shape[:-2]
        for objective in range(self.n_objectives):
            q_obj = qvalues[..., objective]
            all_q_obj = all_qvalues[..., objective]
            mixed_obj = self._mix_single(q_obj, states, one_hot_actions, all_q_obj, available_actions)
            outputs.append(mixed_obj.view(*base_shape))
        return torch.stack(outputs, dim=-1)

    @classmethod
    def from_env(
        cls,
        env: MARLEnv[MultiDiscreteSpace],
        embed_dim: int = 32,
        hypernet_embed: int = 64,
        num_kernel: int = 10,
        adv_hypernet_layers: int = 3,
        adv_hypernet_embed: int = 64,
        is_minus_one: bool = True,
        weighted_head: bool = True,
        is_stop_gradient: bool = True,
    ):
        return cls(
            state_shape=env.state_shape,
            n_agents=env.n_agents,
            n_actions=env.n_actions,
            embed_dim=embed_dim,
            hypernet_embed=hypernet_embed,
            num_kernel=num_kernel,
            adv_hypernet_layers=adv_hypernet_layers,
            adv_hypernet_embed=adv_hypernet_embed,
            is_minus_one=is_minus_one,
            weighted_head=weighted_head,
            is_stop_gradient=is_stop_gradient,
            n_objectives=env.reward_space.size,
        )

    def save(self, to_directory: str):
        filename = f"{to_directory}/qplex.weights"
        torch.save(self.state_dict(), filename)

    def load(self, from_directory: str):
        filename = f"{from_directory}/qplex.weights"
        self.load_state_dict(torch.load(filename, weights_only=True))
