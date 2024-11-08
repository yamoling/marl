from marlenv import MARLEnv
import torch

from marl.models.nn import Mixer


class VDN(Mixer):
    def __init__(self, n_agents: int, multi_objective: bool = False):
        super().__init__(n_agents)
        # The inputs should be of shape (..., n_agents, n_actions)
        if multi_objective:
            self.agent_dim = -2
        self.agent_dim = -1

    def forward(self, qvalues: torch.Tensor, *_args, **_kwargs) -> torch.Tensor:
        # Sum across the agent dimension
        return qvalues.sum(dim=self.agent_dim)

    def save(self, directory: str):
        return

    def load(self, directory: str):
        return

    @classmethod
    def from_env(cls, env: MARLEnv):
        return VDN(env.n_agents, env.is_multi_objective)
