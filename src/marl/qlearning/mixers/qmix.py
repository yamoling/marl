import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QMix(nn.Module):
    def __init__(self, state_shape: int, n_agents: int, embed_size=32, hypernet_embed_size=64):
        super().__init__()

        self.n_agents = n_agents
        self.state_dim = int(np.prod(state_shape))

        self.embed_dim = embed_size

        hypernet_embed = hypernet_embed_size
        self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                        nn.ReLU(),
                                        nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
        self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                        nn.ReLU(),
                                        nn.Linear(hypernet_embed, self.embed_dim))

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs: torch.Tensor, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = torch.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot
