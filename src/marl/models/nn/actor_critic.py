from abc import abstractmethod
from dataclasses import KW_ONLY, dataclass

import torch

from .nn import NN


@dataclass
class Actor[T: torch.distributions.Distribution](NN):
    @abstractmethod
    def policy(
        self,
        obs: torch.Tensor,
        extras: torch.Tensor,
        available_actions: torch.Tensor | None = None,
    ) -> T:
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

    def log_probs(self, obs: torch.Tensor, extras: torch.Tensor, actions: torch.Tensor):
        dist = self.policy(obs, extras, torch.ones_like(actions, dtype=torch.bool))
        return dist.log_prob(actions)


@dataclass
class Critic(NN):
    """Critic neural network"""

    @abstractmethod
    def value(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        """
        Returns a tensor of shape (*dims, n_agents) which represents the value of the observation according to each agent.
        """


@dataclass
class ActorCritic[T: torch.distributions.Distribution](Actor[T], Critic):
    @property
    @abstractmethod
    def policy_parameters(self) -> list[torch.nn.Parameter]:
        pass

    @property
    @abstractmethod
    def value_parameters(self) -> list[torch.nn.Parameter]:
        pass

    @property
    def shared_parameters(self) -> list[torch.nn.Parameter]:
        return []

    def forward(
        self,
        obs: torch.Tensor,
        extras: torch.Tensor,
        available_actions: torch.Tensor | None = None,
    ) -> tuple[T, torch.Tensor]:
        """Returns the logits of the policy distribution and the value function given an observation"""
        return self.policy(obs, extras, available_actions), self.value(obs, extras)


@dataclass
class DiscreteActor(Actor[torch.distributions.Categorical]):
    """Discrete actor neural network"""

    _: KW_ONLY
    clip_logits_low: float | None = None
    clip_logits_high: float | None = None

    @abstractmethod
    def logits(self, obs: torch.Tensor, extras: torch.Tensor, available_actions: torch.Tensor | None = None) -> torch.Tensor:
        """Returns the logits of the policy distribution (clipped if necessary)"""

    def mask(self, x: torch.Tensor, mask: torch.Tensor, replacement=-torch.inf) -> torch.Tensor:
        """Masks the tensor `t` with the boolean tensor `mask`"""
        x[~mask] = replacement
        return x

    def policy(self, obs: torch.Tensor, extras: torch.Tensor, available_actions: torch.Tensor | None = None):
        logits = self.logits(obs, extras, available_actions)
        return torch.distributions.Categorical(logits=logits)

    def to_one_hot(self):
        class DiscreteOneHotActor(Actor[torch.distributions.OneHotCategorical]):
            def __init__(self, actor: DiscreteActor):
                super().__init__(actor.output_shape)
                self.actor = actor

            def __hash__(self):
                return hash(self.name)

            def policy(self, obs: torch.Tensor, extras: torch.Tensor, available_actions: torch.Tensor | None = None):
                logits = self.actor.logits(obs, extras, available_actions)
                return torch.distributions.OneHotCategorical(logits=logits)

        return DiscreteOneHotActor(self)


@dataclass
class DiscreteActorCritic(ActorCritic, DiscreteActor):
    pass
