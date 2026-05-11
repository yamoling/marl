import logging
import math
from dataclasses import KW_ONLY, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from marlenv import DiscreteMARLEnv

from marl.env import EnvConfig
from marl.logging import warn_once
from marl.models.nn import StateMixer
from marl.nn.layers import AbsLayer


@dataclass
class QMix(StateMixer):
    """
    QMix: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning. Supports multiple objectives.

    **Notes**:
        - Paper: https://proceedings.mlr.press/v80/rashid18a/rashid18a.pdf
        - Code inspired from https://github.com/oxwhirl/pymarl
        - In theory, multi-objective should not work with QMIX. Paper: https://openreview.net/pdf?id=NkRZaT2eAk
    """

    _: KW_ONLY
    embed_size: int = 64
    hypernet_embed_size: int = 64

    def __post_init__(self):
        super().__post_init__()
        if self.n_objectives > 1:
            logging.warning("QMIX should not work with multiple objective. See paper at https://openreview.net/pdf?id=NkRZaT2eAk")
        self.hyper_w_1 = nn.Sequential(
            nn.Linear(self.input_size, self.hypernet_embed_size),
            nn.ReLU(),
            nn.Linear(self.hypernet_embed_size, self.embed_size * self.n_agents),
            AbsLayer(),
        )
        self.hyper_w_final = nn.Sequential(
            nn.Linear(self.input_size, self.hypernet_embed_size),
            nn.ReLU(),
            nn.Linear(self.hypernet_embed_size, self.embed_size),
            AbsLayer(),
        )

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.input_size, self.embed_size)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(
            nn.Linear(self.input_size, self.embed_size),
            nn.ReLU(),
            nn.Linear(self.embed_size, 1),
        )

    @property
    def input_size(self):
        return self.state_size + self.state_extras_size

    def forward(
        self,
        qvalues: torch.Tensor,
        states: torch.Tensor,
        states_extras: torch.Tensor,
        /,
        **kwargs,
    ) -> torch.Tensor:
        if len(kwargs) > 0:
            warn_once(f"QMIX received {len(kwargs)} unused keyword arguments: {list(kwargs.keys())}")
        q_totals = []
        batch_dims = states.shape[:-1]
        bs = math.prod(batch_dims)
        states = states.reshape(bs, -1)
        states_extras = states_extras.reshape(bs, -1)
        inputs = torch.cat([states, states_extras], dim=1)
        for i in range(self.n_objectives):
            if self.n_objectives == 1:
                qvalues_obj = qvalues.view(-1, 1, self.n_agents)
            else:
                qvalues_obj = qvalues[:, :, i].view(-1, 1, self.n_agents)
            # First layer
            weight_1 = self.hyper_w_1.forward(inputs)
            bias_1 = self.hyper_b_1.forward(inputs)
            weight_1 = weight_1.view(-1, self.n_agents, self.embed_size)
            bias_1 = bias_1.view(-1, 1, self.embed_size)
            hidden = F.elu(torch.bmm(qvalues_obj, weight_1) + bias_1)
            # Second layer
            weight_2 = self.hyper_w_final.forward(inputs)
            weight_2 = weight_2.view(-1, self.embed_size, 1)
            # State-dependent bias
            value = self.V.forward(inputs).view(-1, 1, 1)
            # Compute final output
            y = torch.bmm(hidden, weight_2) + value
            y = torch.reshape(y, batch_dims)
            q_totals.append(y)
        return torch.stack(q_totals, dim=-1).squeeze()

    @classmethod
    def from_env(cls, env: DiscreteMARLEnv | EnvConfig, embed_size: int = 64, hypernet_embed_size: int = 64, **kwargs):
        return super().from_env(env, hypernet_embed_size=hypernet_embed_size, embed_size=embed_size, **kwargs)

    def __hash__(self):
        return id(self)


@dataclass
class QMixMAVEN(QMix):
    noise_size: int

    @property
    def input_size(self):
        return super().input_size + self.noise_size

    def forward(
        self,
        qvalues: torch.Tensor,
        states: torch.Tensor,
        states_extras: torch.Tensor,
        *,
        maven_noise: torch.Tensor,
        **kwargs,
    ):
        states_extras = torch.cat([states_extras, maven_noise], dim=-1)
        return super().forward(qvalues, states, states_extras, **kwargs)

    @classmethod
    def from_env(cls, env: DiscreteMARLEnv | EnvConfig, embed_size: int = 64, hypernet_embed_size: int = 64, **kwargs):
        noise_size = len([m for m in env.extras_meanings if "maven" in m.lower()])
        if noise_size == 0:
            raise ValueError("No space for the maven noise detected in the environment extras (i.e. with label 'maven').")
        return super().from_env(env, noise_size=noise_size, hypernet_embed_size=hypernet_embed_size, embed_size=embed_size, **kwargs)

    def __hash__(self):
        return id(self)
