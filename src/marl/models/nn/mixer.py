from dataclasses import dataclass
from abc import abstractmethod
import torch


from .nn import NN


@dataclass
class Mixer(NN):
    n_agents: int

    def __init__(self, n_agents: int, n_objectives = 1):
        super().__init__((n_agents,), (0,), (1,))
        self.n_agents = n_agents
        if n_objectives == 1: self.agent_dim = -1
        else: self.agent_dim = -2

    @abstractmethod
    def forward(self, qvalues: torch.Tensor, states: torch.Tensor, **kwargs) -> torch.Tensor:
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
