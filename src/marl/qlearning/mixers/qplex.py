import torch
import math

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
    ):
        super().__init__(n_agents)
        self.n_heads = n_heads
        self.n_actions = n_actions
        self.state_size = state_size
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

    # def _transformation(self, qvalues: torch.Tensor, states: torch.Tensor):
    #     """First step described in the paper is called 'transformation'"""
    #     w, b = self._get_weights_and_biases(states)
    #     qvalues = qvalues * w + b
    #     values = torch.max(qvalues, dim=-1, keepdim=True).values
    #     advantages = qvalues - values
    #     return values, advantages

    # def _duelling_mixing(self, values: torch.Tensor, advantages: torch.Tensor, states: torch.Tensor):
    #     lambdas = self._get_lambdas(states)
    #     values = torch.sum(values, dim=-1)
    #     total_advantages = torch.dot(advantages, lambdas)
    #     return values + total_advantages

    def forward(
        self,
        qvalues: torch.Tensor,
        states: torch.Tensor,
        one_hot_actions: torch.Tensor,
        all_qvalues: torch.Tensor,
    ) -> torch.Tensor:
        states = states.view(-1, self.state_size)
        one_hot_actions = one_hot_actions.view(-1, self.n_actions * self.n_agents)
        qvalues = qvalues.view(-1, self.n_agents)
        all_qvalues = all_qvalues.view(-1, self.n_agents, self.n_actions)
        state_actions = torch.cat([states, one_hot_actions], dim=-1)

        agents = torch.stack([k_ext(states) for k_ext in self.agents_extractors], dim=1)
        actions = torch.stack([sel_ext(state_actions) for sel_ext in self.action_extractors], dim=1)
        keys = torch.stack([k_ext(states) for k_ext in self.key_extractors], dim=1)
        keys = keys.repeat(1, 1, self.n_agents)

        attention = keys * agents * actions
        attention = torch.sum(attention, dim=1)  # sum over heads

        values = torch.max(all_qvalues, dim=-1).values
        advantage = qvalues - values
        advantage = advantage * attention
        advantage = torch.sum(advantage, dim=-1)
        v_tot = torch.sum(qvalues, dim=-1)
        return v_tot + advantage
