from typing import Optional
from dataclasses import dataclass
from typing import Literal
from rlenv import Observation
from abc import ABC, abstractmethod
import torch
from serde import serde

from rlenv.models import RLEnv


def randomize(init_fn, nn: torch.nn.Module):
    for param in nn.parameters():
        if len(param.data.shape) < 2:
            init_fn(param.data.view(1, -1))
        else:
            init_fn(param.data)


@serde
@dataclass
class NN(torch.nn.Module, ABC):
    """Parent class of all neural networks"""

    input_shape: tuple[int, ...]
    extras_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    name: str

    def __init__(self, input_shape: tuple[int, ...], extras_shape: tuple[int, ...], output_shape: tuple[int, ...]):
        torch.nn.Module.__init__(self)
        self.input_shape = tuple(input_shape)
        self.extras_shape = tuple(extras_shape)
        self.output_shape = tuple(output_shape)
        self.name = self.__class__.__name__

    @abstractmethod
    def forward(self, *args):
        """Forward pass"""

    @property
    def device(self) -> torch.device:
        """Returns the device of the model"""
        return next(self.parameters()).device

    def randomize(self, method: Literal["xavier", "orthogonal"] = "xavier"):
        match method:
            case "xavier":
                randomize(torch.nn.init.xavier_uniform_, self)
            case "orthogonal":
                randomize(torch.nn.init.orthogonal_, self)
            case _:
                raise ValueError(f"Unknown initialization method: {method}. Choose between 'xavier' and 'orthogonal'")

    @classmethod
    def from_env(cls, env: RLEnv):
        """Construct a NN from environment specifications"""
        return cls(input_shape=env.observation_shape, extras_shape=env.extra_feature_shape, output_shape=(env.n_actions,))

    def __hash__(self) -> int:
        # Required for deserialization (in torch.nn.module)
        return hash(self.__class__.__name__)


class RecurrentNN(NN):
    def __init__(self, input_shape: tuple[int, ...], extras_shape: tuple[int, ...], output_shape: tuple[int, ...]):
        super().__init__(input_shape, extras_shape, output_shape)
        self.hidden_states: Optional[torch.Tensor] = None
        self.saved_hidden_states = None

    def reset_hidden_states(self):
        """Reset the hidden states"""
        self.hidden_states = None

    def train(self, mode: bool = True):
        if not mode:
            # Set test mode: save training hidden states
            self.saved_hidden_states = self.hidden_states
            self.hidden_states = None
        else:
            # Set train mode
            if not self.training:
                # if not already in train mode, restore hidden states
                self.hidden_states = self.saved_hidden_states
        return super().train(mode)


class QNetwork(NN):
    def to_tensor(self, obs: Observation) -> tuple[torch.Tensor, torch.Tensor]:
        extras = torch.from_numpy(obs.extras).unsqueeze(0).to(self.device)
        obs_tensor = torch.from_numpy(obs.data).unsqueeze(0).to(self.device)
        return obs_tensor, extras

    def qvalues(self, obs: Observation) -> torch.Tensor:
        """Compute the Q-values"""
        return self.forward(*self.to_tensor(obs)).squeeze(0)

    def value(self, obs: Observation) -> torch.Tensor:
        """Compute the value function"""
        agent_values = self.qvalues(obs).max(dim=-1).values
        return agent_values.mean(dim=-1)

    @abstractmethod
    def forward(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        """Compute the Q-values"""

    def batch_forward(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        """Compute the Q-values for a batch of observations during training"""
        return self.forward(obs, extras)

    @classmethod
    def from_env(cls, env: RLEnv):
        return cls(input_shape=env.observation_shape, extras_shape=env.extra_feature_shape, output_shape=(env.n_actions,))


class RecurrentQNetwork(QNetwork, RecurrentNN):
    def batch_forward(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        """
        Compute the Q-values for a batch of observations (multiple episodes) during training.

        In this case, the RNN considers hidden states=None.
        """
        saved_hidden_states = self.hidden_states
        self.hidden_states = None
        qvalues = self.forward(obs, extras)
        self.hidden_states = saved_hidden_states
        return qvalues


class ActorCriticNN(NN, ABC):
    """Actor critic neural network"""

    @property
    @abstractmethod
    def policy_parameters(self) -> list[torch.nn.Parameter]:
        pass

    @property
    @abstractmethod
    def value_parameters(self) -> list[torch.nn.Parameter]:
        pass

    @abstractmethod
    def policy(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns the logits of the policy distribution"""

    @abstractmethod
    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns the value function of an observation"""

    def forward(self, obs: torch.Tensor, extras: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the logits of the policy distribution and the value function given an observation"""
        return self.policy(obs), self.value(obs)


@dataclass(eq=False)
class Mixer(NN, ABC):
    n_agents: int

    def __init__(self, n_agents: int):
        super().__init__((n_agents,), (0,), (1,))
        self.n_agents = n_agents

    @abstractmethod
    def forward(self, qvalues: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Mix the utiliy values of the agents."""

    def save(self, to_directory: str):
        """Save the mixer to a directory."""
        filename = f"{to_directory}/mixer.weights"
        torch.save(self.state_dict(), filename)

    def load(self, from_directory: str):
        """Load the mixer from a directory."""
        filename = f"{from_directory}/mixer.weights"
        self.load_state_dict(torch.load(filename))
