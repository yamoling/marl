from marlenv import MARLEnv
import torch

from marl.models.nn import Mixer


class VDN(Mixer):
    def __init__(self, n_agents: int):
        super().__init__(n_agents)
        # The inputs should be of shape (..., n_agents, n_actions)
        self.agent_dim = -2

    def forward(self, qvalues: torch.Tensor, *_args, **_kwargs) -> torch.Tensor:
        # Sum across the agent dimension
        return qvalues.sum(dim=self.agent_dim)

    def save(self, directory: str):
        return

    def load(self, directory: str):
        return

    @classmethod
    def from_env(cls, env: MARLEnv):
        return VDN(env.n_agents)
