from typing_extensions import Self
from typing import Optional
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar
from abc import ABC, abstractmethod
import torch

from rlenv.models import RLEnv
from marl.utils import Serializable

O = TypeVar("O")

@dataclass(repr=False)
class NN(torch.nn.Module, Serializable, ABC, Generic[O]):
    """Parent class of all neural networks"""
    input_shape: tuple[int, ...]
    extras_shape: Optional[tuple[int, ...]]
    output_shape: tuple[int, ...]

    def __init__(self, input_shape: tuple[int, ...], extras_shape: Optional[tuple[int, ...]], output_shape: tuple[int, ...]):
        torch.nn.Module.__init__(self)
        Serializable.__init__(self)
        self.input_shape = tuple(input_shape)
        self.extras_shape = tuple(extras_shape)
        self.output_shape = tuple(output_shape)
        

    @abstractmethod
    def forward(self, x: torch.Tensor) -> O:
        """Forward pass"""

    @property
    def name(self) -> str:
        return self.__class__.__name__
    
    @property
    def device(self) -> torch.device:
        """Returns the device of the model"""
        return next(self.parameters()).device
    
    def randomize(self, method: Literal["xavier", "orthogonal"]="xavier"):
        match method:
            case "xavier": init = torch.nn.init.xavier_uniform_
            case "orthogonal": init = torch.nn.init.orthogonal_
            case _: raise ValueError(f"Unknown initialization method: {method}. Choose between 'xavier' and 'orthogonal'")
        for param in self.parameters():
            if len(param.data.shape) < 2:
                init(param.data.view(1, -1))
            else:
                init(param.data)
    
    @classmethod
    def from_env(cls, env: RLEnv):
        """Construct a NN from environment specifications"""
        return cls(
            input_shape=env.observation_shape,
            extras_shape=env.extra_feature_shape,
            output_shape=(env.n_actions, )
        )

    def as_dict(self) -> dict[str, ]:
        return {
            **super().as_dict(),
            "layers": str(self)
        }
    
    def __hash__(self) -> int:
        # Required for deserialization (in torch.nn.module)
        return self.__class__.__name__.__hash__()

    @classmethod
    def from_dict(cls, summary: dict[str, ]) -> Self:
        try: summary.pop("layers")
        except KeyError: pass
        return super().from_dict(summary)

class LinearNN(NN[torch.Tensor]):
    """Abstract class defining a linear neural network"""
    @abstractmethod
    def forward(self, obs: torch.Tensor, extras: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass"""

class RecurrentNN(NN[tuple[torch.Tensor, torch.Tensor]]):
    """Abstract class representing a recurrent neural network"""
    @abstractmethod
    def forward(
        self,
        obs: torch.Tensor,
        extras: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None
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
