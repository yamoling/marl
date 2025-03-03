import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from marl.models import Mixer


class QattenMixer(Mixer):
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
        self.state_dim = state_size
        self.u_dim = agent_state_size
        self.n_attention_head = n_heads
        self.mixer_embedding_dim = mixer_embedding_dim
        self.weighted = weighted_head

        self.query_embedding_layers = nn.ModuleList()
        self.key_embedding_layers = nn.ModuleList()
        for _ in range(n_heads):
            self.query_embedding_layers.append(
                nn.Sequential(
                    nn.Linear(self.state_dim, hypernetwork_embed_size),
                    nn.ReLU(),
                    nn.Linear(hypernetwork_embed_size, mixer_embedding_dim),
                )
            )
            self.key_embedding_layers.append(nn.Linear(agent_state_size, mixer_embedding_dim))

        self.scaled_product_value = np.sqrt(mixer_embedding_dim)

        self.head_embedding_layer = nn.Sequential(
            nn.Linear(self.state_dim, mixer_embedding_dim),
            nn.ReLU(),
            nn.Linear(mixer_embedding_dim, n_heads),
        )

        self.constrant_value_layer = nn.Sequential(
            nn.Linear(self.state_dim, self.mixer_embedding_dim), nn.ReLU(), nn.Linear(self.mixer_embedding_dim, 1)
        )

    def forward(self, agent_qs: th.Tensor, states: th.Tensor):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        us = states[:, : self.u_dim * self.n_agents].reshape(-1, self.u_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        q_lambda_list = []
        for i in range(self.n_attention_head):
            state_embedding = self.query_embedding_layers[i](states)
            u_embedding = self.key_embedding_layers[i](us)

            # shape: [-1, 1, state_dim]
            state_embedding = state_embedding.reshape(bs, 1, -1)
            # shape: [-1, state_dim, n_agent]
            u_embedding = u_embedding.reshape(bs, self.n_agents, -1)
            u_embedding = u_embedding.permute(0, 2, 1)

            # shape: [-1, 1, n_agent]
            raw_lambda = th.matmul(state_embedding, u_embedding) / self.scaled_product_value
            q_lambda = F.softmax(raw_lambda, dim=-1)

            q_lambda_list.append(q_lambda)

        # shape: [batch, n_attention_head, n_agent]
        q_lambda_list = th.stack(q_lambda_list, dim=1).squeeze(-2)

        # shape: [batch, n_agent, n_attention_head]
        q_lambda_list = q_lambda_list.permute(0, 2, 1)

        # shape: [batch, 1, n_attention_head]
        q_h = th.matmul(agent_qs, q_lambda_list)

        if self.weighted:
            # shape: [-1, n_attention_head, 1]
            w_h = th.abs(self.head_embedding_layer(states))
            w_h = w_h.reshape(-1, self.n_attention_head, 1)

            # shape: [-1, 1]
            sum_q_h = th.matmul(q_h, w_h)
        else:
            # shape: [-1, 1]
            sum_q_h = q_h.sum(-1)
        sum_q_h = sum_q_h.reshape(-1, 1)

        c = self.constrant_value_layer(states)
        q_tot = sum_q_h + c
        q_tot = q_tot.view(bs, -1)
        return q_tot
