import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .mixer import Mixer


class QMix(Mixer):
    def __init__(self, state_shape: int, n_agents: int, embed_size=32, hypernet_embed_size=64):
        super().__init__(n_agents)

        self.state_shape = state_shape
        self.n_agents = n_agents
        self.embed_size = embed_size
        self.hypernet_embed_size = hypernet_embed_size

        self.state_dim = int(np.prod(state_shape))

        self.hyper_w_1 = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed_size), nn.ReLU(), nn.Linear(hypernet_embed_size, self.embed_size * self.n_agents)
        )
        self.hyper_w_final = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed_size), nn.ReLU(), nn.Linear(hypernet_embed_size, self.embed_size)
        )

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_size)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_size), nn.ReLU(), nn.Linear(self.embed_size, 1))

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1.forward(states)
        w1 = w1.view(-1, self.n_agents, self.embed_size)
        b1 = b1.view(-1, 1, self.embed_size)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_size, 1)
        # State-dependent bias
        v = self.V.forward(states).view(-1, 1, 1)
        # Compute final output
        y = torch.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1)
        return q_tot

    def save(self, to_directory: str):
        filename = f"{to_directory}/qmix.model"
        torch.save(self.state_dict(), filename)

    def load(self, from_directory: str):
        filename = f"{from_directory}/qmix.model"
        return self.load_state_dict(torch.load(filename))
