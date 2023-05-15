from typing_extensions import Self
from typing import Generic, TypeVar
from abc import ABC, abstractmethod
import torch

from rlenv.models import RLEnv
from marl.utils.summarizable import Summarizable

O = TypeVar("O")

class NN(torch.nn.Module, Summarizable, ABC, Generic[O]):
    """Parent class of all neural networks"""
    is_recurrent: bool

    def __init__(self, input_shape: tuple[int, ...], extras_shape: tuple[int, ...]|None, output_shape: tuple[int, ...]) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.extras_shape = extras_shape
        self.output_shape = output_shape

    @abstractmethod
    def forward(self, x: torch.Tensor) -> O:
        """Forward pass"""

    @property
    def is_recurrent(self) -> bool:
        """Returns whether the model is recurrent or not"""
        return False
    
    @property
    def name(self) -> str:
        return self.__class__.__name__
    
    def randomized(self) -> Self:
        for param in self.parameters():
            if len(param.data.shape) == 1:
                torch.nn.init.xavier_uniform_(param.data.view(1, -1))
            else:
                torch.nn.init.xavier_uniform_(param.data)
        return self
    
    @classmethod
    def from_env(cls, env: RLEnv):
        """Construct a NN from environment specifications"""
        return cls(
            input_shape=env.observation_shape,
            extras_shape=env.extra_feature_shape,
            output_shape=(env.n_actions, )
        )

    def summary(self) -> dict[str, ]:
        return {
            **super().summary(),
            "input_shape": self.input_shape,
            "extras_shape": self.extras_shape,
            "output_shape": self.output_shape,
            "layers": str(self)
        }
    
    @classmethod
    def from_summary(cls, summary: dict[str, ]) -> Self:
        try: summary.pop("layers")
        except KeyError: pass
        return super().from_summary(summary)

class LinearNN(NN[torch.Tensor], ABC):
    """Abstract class defining a linear neural network"""
    @abstractmethod
    def forward(self, obs: torch.Tensor, extras: torch.Tensor|None = None) -> torch.Tensor:
        """Forward pass"""

class RecurrentNN(NN[tuple[torch.Tensor, torch.Tensor]], ABC):
    """Abstract class representing a recurrent neural network"""
    @abstractmethod
    def forward(
        self,
        obs: torch.Tensor,
        extras: torch.Tensor|None = None,
        hidden_states: torch.Tensor|None = None
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

    @property
    def is_recurrent(self) -> bool:
        return True


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
