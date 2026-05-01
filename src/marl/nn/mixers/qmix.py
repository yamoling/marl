from dataclasses import KW_ONLY, dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from marlenv import MARLEnv, MultiDiscreteSpace

from marl.models.nn import Mixer
from marl.nn.layers import AbsLayer


@dataclass(unsafe_hash=True)
class QMix(Mixer):
    """
    QMix: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning

    (almost) copy-pasted from https://github.com/oxwhirl/pymarl
    """

    state_size: int
    state_extras_size: int
    _: KW_ONLY
    embed_size: int = 64
    hypernet_embed_size: int = 64

    def __post_init__(self):
        super().__post_init__()

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
        self.hyper_b_1 = nn.Linear(
            self.input_size,
            self.embed_size,
        )

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
        maven_noise: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        q_totals = []
        batch_dims = states.shape[:-1]
        bs = math.prod(batch_dims)
        states = states.reshape(bs, -1)
        states_extras = states_extras.reshape(bs, -1)
        inputs = [states, states_extras]
        if maven_noise is not None:
            inputs.append(maven_noise.reshape(bs, -1))
        inputs = torch.cat(inputs, dim=1)
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
    def from_env(
        cls,
        env: MARLEnv[MultiDiscreteSpace],
        embed_size: int = 64,
        hypernet_embed_size: int = 64,
        maven_noise_size: int | None = None,
    ):
        state_size = env.state_size
        if maven_noise_size is not None:
            state_size += maven_noise_size
        return QMix(
            (env.n_objectives,),
            env.n_agents,
            state_size,
            env.state_extras_size,
            embed_size=embed_size,
            hypernet_embed_size=hypernet_embed_size,
            n_objectives=env.n_objectives,
        )

    def save(self, to_directory: str):
        filename = f"{to_directory}/qmix.weights"
        torch.save(self.state_dict(), filename)

    def load(self, from_directory: str):
        filename = f"{from_directory}/qmix.weights"
        self.load_state_dict(torch.load(filename, weights_only=True))
