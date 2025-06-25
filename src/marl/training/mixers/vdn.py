from dataclasses import dataclass

import torch
from marlenv import MARLEnv

from marl.models.nn import Mixer


@dataclass(unsafe_hash=True)
class VDN(Mixer):
    def __init__(self, n_agents: int, n_objectives = 1):
        super().__init__(n_agents, n_objectives)

    def forward(self, qvalues: torch.Tensor, *_args, **_kwargs) -> torch.Tensor:
        # Sum across the agent dimension
        return torch.sum(qvalues, dim = self.agent_dim)

    def save(self, directory: str):
        return

    def load(self, directory: str):
        return

    @classmethod
    def from_env[A](cls, env: MARLEnv[A]):
        return VDN(env.n_agents, env.reward_space.size)
