from abc import abstractmethod
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path

import torch
from marlenv import DiscreteMARLEnv

from marl.env import EnvConfig

from .nn import NN


@dataclass
class Mixer(NN):
    output_shape: tuple[int, ...] = field(init=False)
    _: KW_ONLY
    n_objectives: int = 1

    def __post_init__(self):
        super().__post_init__()
        self.output_shape = (self.n_objectives,)

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def agent_dim(self):
        if self.n_objectives == 1:
            return -1
        return -2

    @abstractmethod
    def forward(self, qvalues: torch.Tensor, states: torch.Tensor, states_extras: torch.Tensor, /, **kwargs) -> torch.Tensor:
        """
        Mix the utiliy values of the agents.

        To englobe every possible mixer, the signature of the forward method is quite complex.
        - qvalues: the Q-values of the action take by each agent. (batch, n_agents)
        - states: the state of the environment. (batch, state_size)
        """

    def save(self, directory: Path):
        """Save the mixer to a directory."""
        filename = f"{directory}/mixer.weights"
        state_dict = self.state_dict()
        if len(state_dict) == 0:
            return
        torch.save(state_dict, filename)

    @classmethod
    def from_env(cls, env: DiscreteMARLEnv | EnvConfig[DiscreteMARLEnv], **kwargs):
        """Create a mixer from an environment."""
        return cls(n_objectives=env.n_objectives, **kwargs)


@dataclass
class StateMixer(Mixer):
    n_agents: int
    state_size: int
    state_extras_size: int

    @classmethod
    def from_env(cls, env: DiscreteMARLEnv | EnvConfig[DiscreteMARLEnv], **kwargs):
        return super().from_env(env, n_agents=env.n_agents, state_size=env.state_size, state_extras_size=env.state_extras_size, **kwargs)
