from dataclasses import KW_ONLY, dataclass

import torch
from torch import nn

from marl.models.nn import StateMixer
from marl.nn.layers import AbsLayer


@dataclass
class Qatten(StateMixer):
    state_size: int
    state_extras_size: int
    unit_dim: int
    _: KW_ONLY
    mixer_embedding_dim: int = 32
    hypernetwork_embed_size: int = 64
    n_heads: int = 4
    weighted_head: bool = False

    @property
    def input_size(self):
        return self.state_size + self.state_extras_size

    def __hash__(self):
        return id(self)

    def __post_init__(self):
        super().__post_init__()

        self.value = nn.Sequential(
            nn.Linear(self.input_size, self.mixer_embedding_dim),
            nn.ReLU(),
            nn.Linear(self.mixer_embedding_dim, 1),
        )

        self.hyper_w_head = nn.Sequential(
            nn.Linear(self.input_size, self.mixer_embedding_dim),
            nn.ReLU(),
            nn.Linear(self.mixer_embedding_dim, self.n_heads),
            AbsLayer(),
        )
        self.key_extractors = nn.ModuleList()
        self.query_extractors = nn.ModuleList()
        for _ in range(self.n_heads):  # Manual implementation of multi-head attention
            self.query_extractors.append(
                nn.Sequential(
                    nn.Linear(self.input_size, self.hypernetwork_embed_size),
                    nn.ReLU(),
                    nn.Linear(self.hypernetwork_embed_size, self.mixer_embedding_dim, bias=False),
                )
            )
            self.key_extractors.append(nn.Linear(self.unit_dim, self.mixer_embedding_dim, bias=False))  # key

    def forward(self, qvalues: torch.Tensor, states: torch.Tensor, states_extras: torch.Tensor, /, **kwargs):
        """Mix arbitrary leading batch dimensions using scalar attention values. @ai-generated"""
        dims = states.shape[:-1]
        states = states.reshape(-1, self.state_size)
        states_extras = states_extras.reshape(states.shape[0], self.state_extras_size)
        inputs = torch.cat([states, states_extras], dim=1)
        unit_states = states[:, : self.unit_dim * self.n_agents]  # get agent own features from state
        unit_states = unit_states.view(-1, self.n_agents, self.unit_dim)

        # Scalar values compute the weighted sum directly, avoiding an agent-by-agent diagonal matrix.
        values = qvalues.reshape(-1, self.n_agents, 1)
        attentioned_qvalues = []
        for key_extractor, query_extractor in zip(self.key_extractors, self.query_extractors):
            keys = key_extractor(unit_states)  # shape (batch, n_agents, embed_dim)
            queries = query_extractor.forward(inputs)  # shape (batch, embed_dim)
            queries = queries.unsqueeze(-2)  # Appropriate shape for dot product
            head_output = torch.nn.functional.scaled_dot_product_attention(queries, keys, values)
            attentioned_qvalues.append(head_output.squeeze(1))
        attentioned_qvalues = torch.stack(attentioned_qvalues, dim=1)

        # Sum over the agents to get the final $q^h$ as shown in the paper schematic.
        q_h = torch.sum(attentioned_qvalues, dim=-1)

        # If we use weighted heads (right path of the figure), then compute them from the states and apply them to q_h.
        if self.weighted_head:
            w_head = self.hyper_w_head.forward(inputs)
            q_h = q_h * w_head

        # Top side of the figure: add c(s), which is V(s) in practice.
        v = torch.squeeze(self.value.forward(inputs))
        q_sum = torch.sum(q_h, dim=1)
        q_tot = q_sum + v
        return q_tot.view(*dims)
