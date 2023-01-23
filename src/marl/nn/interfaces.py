from typing import Optional, Tuple
from abc import ABC, abstractmethod
import torch

from rlenv.models import RLEnv


class NN(torch.nn.Module, ABC):
    """Parent class of all neural networks"""
    is_recurrent: bool

    def __init__(self, input_shape: tuple[int, ...], extras_shape: tuple[int, ...]|None, output_shape: tuple[int, ...]) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.extras_shape = extras_shape
        self.output_shape = output_shape

    @classmethod
    def from_env(cls, env: RLEnv):
        """Construct a NN from environment specifications"""
        return cls(
            input_shape=env.observation_shape,
            extras_shape=env.extra_feature_shape,
            output_shape=(env.n_actions, )
        )

    @property
    def is_recurrent(self) -> bool:
        """Returns whether the model is recurrent or not"""
        return False

class LinearNN(NN, ABC):
    """Abstract class defining a linear neural network"""
    @abstractmethod
    def forward(self, obs: torch.Tensor, extras: torch.Tensor|None = None) -> torch.Tensor:
        """Forward pass"""

class RecurrentNN(NN, ABC):
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
