from marlenv import MARLEnv, MultiDiscreteSpace
import torch

from torch import nn
from marl.models.nn import Mixer
from marl.nn.layers import AbsLayer


class QPlex(Mixer):
    """Duplex dueling"""

    def __init__(
        self,
        n_agents: int,
        n_actions: int,
        n_heads: int,
        state_size: int,
        adv_hypernet_embed: int,
        weighted_head: bool = True,
    ):
        super().__init__(n_agents)
        self.n_heads = n_heads
        self.n_actions = n_actions
        self.state_size = state_size
        self.weighted_head = weighted_head
        self.key_extractors = nn.ModuleList()
        self.agents_extractors = nn.ModuleList()
        self.action_extractors = nn.ModuleList()

        for _ in range(n_heads):  # multi-head attention
            self.key_extractors.append(
                nn.Sequential(
                    nn.Linear(state_size, adv_hypernet_embed),
                    nn.ReLU(),
                    nn.Linear(adv_hypernet_embed, adv_hypernet_embed),
                    nn.ReLU(),
                    nn.Linear(adv_hypernet_embed, 1),
                    AbsLayer(),
                )
            )  # key
            self.agents_extractors.append(
                nn.Sequential(
                    nn.Linear(state_size, adv_hypernet_embed),
                    nn.ReLU(),
                    nn.Linear(adv_hypernet_embed, adv_hypernet_embed),
                    nn.ReLU(),
                    nn.Linear(adv_hypernet_embed, n_agents),
                    nn.Sigmoid(),
                )
            )  # agent
            self.action_extractors.append(
                nn.Sequential(
                    nn.Linear(state_size + n_agents * n_actions, adv_hypernet_embed),
                    nn.ReLU(),
                    nn.Linear(adv_hypernet_embed, adv_hypernet_embed),
                    nn.ReLU(),
                    nn.Linear(adv_hypernet_embed, n_agents),
                    nn.Sigmoid(),
                )
            )  # action
        self.weights_generator = nn.Sequential(
            nn.Linear(state_size, adv_hypernet_embed),
            nn.ReLU(),
            nn.Linear(adv_hypernet_embed, n_agents),
            AbsLayer(),
        )
        self.V = nn.Sequential(
            nn.Linear(state_size, adv_hypernet_embed),
            nn.ReLU(),
            nn.Linear(adv_hypernet_embed, n_agents),
        )

    def transformation(self, states: torch.Tensor, qvalues: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.weights_generator(states) + 1e-10
        states_value = self.V(states)
        qvalues = qvalues * weights + states_value
        values = values * weights + states_value
        advantages = qvalues - values
        return values, advantages

    def dueling_mixing(
        self,
        advantages: torch.Tensor,
        values: torch.Tensor,
        states: torch.Tensor,
        one_hot_actions: torch.Tensor,
    ) -> torch.Tensor:
        state_actions = torch.cat([states, one_hot_actions], dim=-1)
        # Compute attention weights
        agents = torch.stack([k_ext(states) for k_ext in self.agents_extractors], dim=1)
        actions = torch.stack([sel_ext(state_actions) for sel_ext in self.action_extractors], dim=1)
        keys = torch.stack([k_ext(states) for k_ext in self.key_extractors], dim=1)
        keys = keys.repeat(1, 1, self.n_agents)
        attention = keys * agents * actions
        attention = torch.sum(attention, dim=1)  # sum over heads
        attention = attention - 1  # Don't know why they do this but they do it in the original code

        # Weight the advantage with the attention
        advantage = advantages * attention
        a_tot = torch.sum(advantage, dim=-1)
        v_tot = torch.sum(values, dim=-1)
        q_tot = v_tot + a_tot
        return q_tot

    def forward(
        self,
        qvalues: torch.Tensor,
        states: torch.Tensor,
        one_hot_actions: torch.Tensor,
        all_qvalues: torch.Tensor,
        *_args,
        **_kwargs,
    ) -> torch.Tensor:
        *dims, _ = qvalues.shape
        states = states.reshape(-1, self.state_size)
        one_hot_actions = one_hot_actions.view(-1, self.n_actions * self.n_agents)
        qvalues = qvalues.view(-1, self.n_agents)

        # Value of a state is the maximal qvalue
        values = all_qvalues.max(dim=-1).values
        values = values.view(-1, self.n_agents)

        # Weighted heads
        if self.weighted_head:
            values, advantages = self.transformation(states, qvalues, values)
        else:
            advantages = qvalues - values
        # I don't know why we need to detach the advantages here but they do it in the original code
        # and it does not work without it.
        advantages = advantages.detach()
        q_tot = self.dueling_mixing(advantages, values, states, one_hot_actions)
        return q_tot.view(*dims)

    @classmethod
    def from_env[A](cls, env: MARLEnv[MultiDiscreteSpace], adv_hypernet_embed: int = 64, transformation=True):
        assert len(env.state_shape) == 1
        return QPlex(env.n_agents, env.n_actions, env.state_shape[0], adv_hypernet_embed, transformation)
