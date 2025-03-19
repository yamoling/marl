from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch

from .nn import NN


@dataclass
class ActorNN(NN, ABC):
    def __init__(self, input_shape: tuple[int, ...], extras_shape: tuple[int, ...], action_output_shape: tuple[int, ...]):
        NN.__init__(self, input_shape, extras_shape, action_output_shape)
        self.action_shape = action_output_shape

    @abstractmethod
    def policy(
        self,
        data: torch.Tensor,
        extras: torch.Tensor,
        available_actions: Optional[torch.Tensor] = None,
    ) -> torch.distributions.Distribution:
        """
        Returns the probability distribution over the actions.

        Note that the `available_actions` are only relevant to discrete action spaces.
        The `available_actions` should be a boolean tensor of shape (*dims, n_actions) where `True` means that the action is available.
        The probability of actions that are not avaliable is zero.
        """

    def mask(self, x: torch.Tensor, mask: torch.Tensor, replacement=-torch.inf) -> torch.Tensor:
        """Masks the input tensor `x` with the boolean tensor `mask`"""
        x[~mask] = replacement
        return x


@dataclass
class DiscreteActorNN(ActorNN, ABC):
    """Discrete actor neural network"""

    clip_logits_low: Optional[float]
    clip_logits_high: Optional[float]

    def __init__(
        self,
        input_shape: tuple[int, ...],
        extras_shape: tuple[int, ...],
        action_output_shape: tuple[int, ...],
        clip_logits_low: Optional[float] = None,
        clip_logits_high: Optional[float] = None,
    ):
        ActorNN.__init__(self, input_shape, extras_shape, action_output_shape)
        self.action_output_shape = action_output_shape
        self.clip_logits_low = clip_logits_low
        self.clip_logits_high = clip_logits_high

    @abstractmethod
    def logits(self, data: torch.Tensor, extras: torch.Tensor, available_actions: torch.Tensor) -> torch.Tensor:
        """Returns the logits of the policy distribution (clipped if necessary)"""

    def mask(self, t: torch.Tensor, mask: torch.Tensor, replacement=-torch.inf) -> torch.Tensor:
        """Masks the tensor `t` with the boolean tensor `mask`"""
        t[~mask] = replacement
        return t

    def policy(self, data: torch.Tensor, extras: torch.Tensor, available_actions: torch.Tensor) -> torch.distributions.Distribution:
        logits = self.logits(data, extras, available_actions)
        return torch.distributions.Categorical(logits=logits)


@dataclass
class CriticNN(NN, ABC):
    """Critic neural network"""

    def __init__(self, input_shape: tuple[int, ...], extras_shape: tuple[int, ...]):
        NN.__init__(self, input_shape, extras_shape, (1,))
        self.value_output_shape = (1,)

    @abstractmethod
    def value(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        """Returns the value of an observation (for each agent)"""


@dataclass
class ActorCriticNN(ActorNN, CriticNN):
    def __init__(self, input_shape: tuple[int, ...], extras_shape: tuple[int, ...], action_output_shape: tuple[int, ...]):
        ActorNN.__init__(self, input_shape, extras_shape, action_output_shape)
        CriticNN.__init__(self, input_shape, extras_shape)

    @property
    @abstractmethod
    def policy_parameters(self) -> list[torch.nn.Parameter]:
        pass

    @property
    @abstractmethod
    def value_parameters(self) -> list[torch.nn.Parameter]:
        pass

    def forward(
        self,
        obs: torch.Tensor,
        extras: torch.Tensor,
        available_actions: torch.Tensor,
    ) -> tuple[torch.distributions.Distribution, torch.Tensor]:
        """Returns the logits of the policy distribution and the value function given an observation"""
        return self.policy(obs, extras, available_actions), self.value(obs, extras)


@dataclass
class DiscreteActorCriticNN(ActorCriticNN, DiscreteActorNN):
    def __init__(
        self,
        input_shape: tuple[int, ...],
        extras_shape: tuple[int, ...],
        action_output_shape: tuple[int, ...],
        clip_logits_low: Optional[float] = None,
        clip_logits_high: Optional[float] = None,
    ):
        ActorCriticNN.__init__(self, input_shape, extras_shape, action_output_shape)
        DiscreteActorNN.__init__(self, input_shape, extras_shape, action_output_shape, clip_logits_low, clip_logits_high)
