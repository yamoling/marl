from dataclasses import dataclass

import torch
from marlenv import MultiDiscreteSpace, MARLEnv

from marl.models.nn import Mixer


@dataclass(unsafe_hash=True)
class VDN(Mixer):
    def __init__(self, n_agents: int):
        super().__init__(n_agents)

    def forward(self, qvalues: torch.Tensor, *_args, **_kwargs) -> torch.Tensor:
        # Sum across the agent dimension
        return torch.sum(qvalues, dim=-1)

    def save(self, directory: str):
        return

    def load(self, directory: str):
        return

    @classmethod
    def from_env[A](cls, env: MARLEnv[MultiDiscreteSpace]):
        return VDN(env.n_agents)
