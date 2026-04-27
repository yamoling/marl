from dataclasses import dataclass

import torch
from marlenv import MARLEnv, MultiDiscreteSpace

from marl.models.nn import Mixer


@dataclass(unsafe_hash=True)
class VDN(Mixer):
    def forward(self, qvalues: torch.Tensor, *_args, **_kwargs) -> torch.Tensor:
        # Sum across the agent dimension
        return torch.sum(qvalues, dim=self.agent_dim)

    def save(self, to_directory: str):
        return

    def load(self, from_directory: str):
        return

    @classmethod
    def from_env(cls, env: MARLEnv[MultiDiscreteSpace]):
        return VDN(env.n_agents, n_objectives=env.reward_space.size)
