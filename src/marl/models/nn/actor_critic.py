from typing import Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch


from .nn import NN


@dataclass
class DiscreteActorNN(NN, ABC):
    """Actor neural network"""

    def __init__(self, input_shape: tuple[int, ...], extras_shape: tuple[int, ...], action_output_shape: tuple[int, ...]):
        super().__init__(input_shape, extras_shape, action_output_shape)
        self.action_output_shape = action_output_shape

    @abstractmethod
    def logits(self, data: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        """Returns the logits of the policy distribution"""

    def policy(self, data: torch.Tensor, extras: torch.Tensor, available_actions: torch.Tensor | None = None) -> torch.Tensor:
        """
        Returns the probabilities of the policy distribution, i.e. the softmax of the logits.

        The `available_actions` should be a boolean tensor of shape (*dims, n_actions) where `True` means that the action is available.
        """
        logits = self.logits(data, extras)
        if available_actions is not None:
            logits[~available_actions] = -torch.inf
        return torch.nn.functional.softmax(logits, -1)


@dataclass
class ContinuousActorNN(NN):
    def __init__(self, input_shape: tuple[int, ...], extras_shape: tuple[int, ...], action_output_shape: tuple[int, ...]):
        NN.__init__(self, input_shape, extras_shape, action_output_shape)
        self.action_output_shape = action_output_shape

    @abstractmethod
    def policy(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.distributions.Distribution:
        """Returns the policy distribution over actions"""


@dataclass
class CriticNN(NN, ABC):
    """Critic neural network"""

    def __init__(self, input_shape: tuple[int, ...], extras_shape: tuple[int, ...]):
        super().__init__(input_shape, extras_shape, (1,))
        self.value_output_shape = (1,)

    @abstractmethod
    def value(self, data: torch.Tensor, extras: torch.Tensor, action_probs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns the value function of an observation"""


@dataclass
class DiscreteActorCriticNN(DiscreteActorNN, CriticNN, ABC):
    """Actor critic neural network"""

    def __init__(self, input_shape: tuple[int, ...], extras_shape: tuple[int, ...], n_actions: int):
        DiscreteActorNN.__init__(self, input_shape, extras_shape, (n_actions,))
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
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the logits of the policy distribution and the value function given an observation"""
        return self.policy(obs, extras, action_mask), self.value(obs, extras)


@dataclass
class ContinuousActorCriticNN(ContinuousActorNN, CriticNN, ABC):
    """Actor critic neural network"""

    def __init__(self, input_shape: tuple[int, ...], extras_shape: tuple[int, ...], action_output_shape: tuple[int, ...]):
        ContinuousActorNN.__init__(self, input_shape, extras_shape, action_output_shape)
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
    ):
        """Returns the means and std of the policy distribution and the value function given an observation"""
        return self.policy(obs, extras), self.value(obs, extras)
