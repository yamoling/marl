from abc import abstractmethod
from dataclasses import KW_ONLY, dataclass

import torch

from .nn import NN


@dataclass
class Mixer(NN):
    n_agents: int
    _: KW_ONLY
    n_objectives: int = 1

    def __post_init__(self):
        super().__post_init__()
        if self.n_objectives == 1:
            self.agent_dim = -1
        else:
            self.agent_dim = -2

    @abstractmethod
    def forward(self, qvalues: torch.Tensor, states: torch.Tensor, states_extras: torch.Tensor, /, **kwargs) -> torch.Tensor:
        """
        Mix the utiliy values of the agents.

        To englobe every possible mixer, the signature of the forward method is quite complex.
        - qvalues: the Q-values of the action take by each agent. (batch, n_agents)
        - states: the state of the environment. (batch, state_size)
        """

    def save(self, to_directory: str):
        """Save the mixer to a directory."""
        filename = f"{to_directory}/mixer.weights"
        torch.save(self.state_dict(), filename)

    def load(self, from_directory: str):
        """Load the mixer from a directory."""
        filename = f"{from_directory}/mixer.weights"
        self.load_state_dict(torch.load(filename, weights_only=True))
