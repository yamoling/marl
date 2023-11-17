from typing import Optional
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar
from abc import ABC, abstractmethod
import torch
from serde import serde

from rlenv.models import RLEnv

Output = TypeVar("Output")


def randomize(init_fn, nn: torch.nn.Module):
    for param in nn.parameters():
        if len(param.data.shape) < 2:
            init_fn(param.data.view(1, -1))
        else:
            init_fn(param.data)


@serde
@dataclass
class NN(torch.nn.Module, ABC, Generic[Output]):
    """Parent class of all neural networks"""

    input_shape: tuple[int, ...]
    extras_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    name: str

    def __init__(self, input_shape: tuple[int, ...], extras_shape: Optional[tuple[int, ...]], output_shape: tuple[int, ...]):
        torch.nn.Module.__init__(self)
        self.input_shape = tuple(input_shape)
        if extras_shape is None:
            self.extras_shape = (0,)
        else:
            self.extras_shape = tuple(extras_shape)
        self.output_shape = tuple(output_shape)
        self.name = self.__class__.__name__

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Output:
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
        # for param in self.parameters():
        #     if len(param.data.shape) < 2:
        #         init(param.data.view(1, -1))
        #     else:
        #         init(param.data)

    @classmethod
    def from_env(cls, env: RLEnv):
        """Construct a NN from environment specifications"""
        return cls(input_shape=env.observation_shape, extras_shape=env.extra_feature_shape, output_shape=(env.n_actions,))

    def __hash__(self) -> int:
        # Required for deserialization (in torch.nn.module)
        return hash(self.__class__.__name__)


class LinearNN(NN[torch.Tensor]):
    """Abstract class defining a linear neural network"""

    @abstractmethod
    def forward(self, obs: torch.Tensor, extras: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass"""


class RecurrentNN(NN[tuple[torch.Tensor, torch.Tensor]]):
    """Abstract class representing a recurrent neural network"""

    @abstractmethod
    def forward(
        self, obs: torch.Tensor, extras: Optional[torch.Tensor] = None, hidden_states: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        All the inputs must have the shape [Episode length, Number of agents, *data_shape]

        Arguments:
            - (torch.Tensor) obs: the observation(s)
            - (torch.Tensor) extras: the extra features (agent ID, ...)
            - (torch.Tensor) hidden_states: the hidden states

        Note that obs, extras and hidden shapes must have the same number of dimensions

        Returns:
            - (torch.Tensor) The NN output
            - (torch.Tensor) The hidden states
        """


class ActorCriticNN(NN[tuple[torch.Tensor, torch.Tensor]], ABC):
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
