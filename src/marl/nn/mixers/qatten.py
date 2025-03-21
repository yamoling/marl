import torch.nn as nn
import torch
from marl.models import Mixer
from marl.nn.layers import AbsLayer


class Qatten(Mixer):
    def __init__(
        self,
        n_agents: int,
        state_size: int,
        agent_state_size: int,
        mixer_embedding_dim: int = 32,
        hypernetwork_embed_size: int = 64,
        n_heads: int = 4,
        weighted_head: bool = False,
    ):
        super().__init__(n_agents)

        self.n_agents = n_agents
        self.unit_dim = agent_state_size
        self.weighted_head = weighted_head

        self.value = nn.Sequential(
            nn.Linear(state_size, mixer_embedding_dim),
            nn.ReLU(),
            nn.Linear(mixer_embedding_dim, 1),
        )

        self.hyper_w_head = nn.Sequential(
            nn.Linear(state_size, mixer_embedding_dim),
            nn.ReLU(),
            nn.Linear(mixer_embedding_dim, n_heads),
            AbsLayer(),
        )
        self.key_extractors = nn.ModuleList()
        self.query_extractors = nn.ModuleList()
        for _ in range(n_heads):  # Manual implementation of multi-head attention
            self.query_extractors.append(
                nn.Sequential(
                    nn.Linear(state_size, hypernetwork_embed_size),
                    nn.ReLU(),
                    nn.Linear(hypernetwork_embed_size, mixer_embedding_dim, bias=False),
                )
            )
            self.key_extractors.append(nn.Linear(agent_state_size, mixer_embedding_dim, bias=False))  # key

    def forward(
        self,
        qvalues: torch.Tensor,
        states: torch.Tensor,
    ):
        *dims, state_size = states.shape
        states = states.reshape(-1, state_size)
        unit_states = states[:, : self.unit_dim * self.n_agents]  # get agent own features from state
        unit_states = unit_states.view(-1, self.n_agents, self.unit_dim)

        # Computation of $q^h$ in Figure 1 (output of the "middle" dot product).
        # We need a diagonal matrix of the qvalues to multiply with the attention weights. (cf original code
        # that just does a simple multiplication with the qvalues, which is equivalent to a dot product with a
        # diagonal matrix of the qvalues).
        values = qvalues.view(-1, self.n_agents).diag_embed()
        attentioned_qvalues = []
        for key_extractor, query_extractor in zip(self.key_extractors, self.query_extractors):
            keys = key_extractor(unit_states)  # shape (batch, n_agents, embed_dim)
            queries = query_extractor(states)  # shape (batch, embed_dim)
            queries = queries.unsqueeze(-2)  # Appropriate shape for dot product
            head_output = torch.nn.functional.scaled_dot_product_attention(queries, keys, values)
            attentioned_qvalues.append(head_output.squeeze(1))
        attentioned_qvalues = torch.stack(attentioned_qvalues, dim=1)

        # Sum over the agents to get the final $q^h$ as shown in the paper schematic.
        q_h = torch.sum(attentioned_qvalues, dim=-1)

        # If we use weighted heads (right path of the figure), then compute them from the states and apply them to q_h.
        if self.weighted_head:
            w_head = self.hyper_w_head.forward(states)
            q_h = q_h * w_head

        # Top side of the figure: add c(s), which is V(s) in practice.
        v = torch.squeeze(self.value.forward(states))
        q_sum = torch.sum(q_h, dim=1)
        q_tot = q_sum + v
        return q_tot.view(*dims)
