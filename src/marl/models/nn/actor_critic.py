import math
from abc import abstractmethod
from dataclasses import KW_ONLY, dataclass, field

import torch
from marlenv import ContinuousSpace, MARLEnv
from torch.distributions import transforms

from marl.env import EnvConfig

from .nn import NN


@dataclass
class Actor[T: torch.distributions.Distribution](NN):
    n_actions: int
    obs_shape: tuple[int, ...]
    extras_shape: tuple[int, ...]
    _: KW_ONLY
    independent: bool = False
    n_agents: int = -1
    output_shape: tuple[int, ...] = field(init=False)

    @abstractmethod
    def policy(self, obs: torch.Tensor, extras: torch.Tensor, *, available_actions: torch.Tensor | None = None, **kwargs) -> T:
        """
        Returns the probability distribution over the actions.

        Note that the `available_actions` are only relevant to discrete action spaces.
        The `available_actions` should be a boolean tensor of shape (*dims, n_actions) where `True` means that the action is available.
        The probability of actions that are not avaliable should be zero.
        """

    @abstractmethod
    def forward(
        self,
        obs: torch.Tensor,
        extras: torch.Tensor,
        *,
        available_actions: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the logits of the distribution."""

    def mask(self, x: torch.Tensor, available_actions: torch.Tensor | None, replacement=-torch.inf):
        """Masks the input tensor `x` with the boolean tensor `mask`"""
        if available_actions is None:
            return x
        return x.masked_fill(~available_actions, replacement)

    def log_probs(self, obs: torch.Tensor, extras: torch.Tensor, actions: torch.Tensor):
        dist = self.policy(obs, extras)
        return dist.log_prob(actions)

    @property
    def obs_size(self):
        return math.prod(self.obs_shape)

    @property
    def extras_size(self):
        return math.prod(self.extras_shape)

    @classmethod
    def from_env(cls, env: EnvConfig | MARLEnv, *, independent: bool = True, **kwargs):
        return cls(
            env.n_actions,
            env.observation_shape,
            env.extras_shape,
            n_agents=env.n_agents,
            independent=independent,
            **kwargs,
        )


@dataclass
class Critic(NN):
    """Critic neural network"""

    obs_shape: tuple[int, ...]
    extras_shape: tuple[int, ...]
    _: KW_ONLY
    output_shape: tuple[int, ...] = field(init=False)
    independent: bool = False
    n_agents: int = 0

    def __post_init__(self):
        if self.independent:
            assert self.n_agents > 0
        self.output_shape = (1,)
        super().__post_init__()

    def value(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        """
        Returns a tensor of shape (*dims, n_agents) that represents the value of the observation according to each agent.
        """
        return self.forward(obs, extras)

    @classmethod
    def from_env(cls, env: EnvConfig | MARLEnv, independent: bool = False, **kwargs):
        return cls(env.observation_shape, env.extras_shape, n_agents=env.n_agents, independent=independent, **kwargs)

    @property
    def obs_size(self):
        return math.prod(self.obs_shape)

    @property
    def extras_size(self):
        return math.prod(self.extras_shape)


@dataclass
class CategoricalActor(Actor[torch.distributions.Categorical]):
    """Discrete actor neural network"""

    def __post_init__(self):
        super().__post_init__()
        self.output_shape = (self.n_actions,)

    def policy(
        self,
        obs: torch.Tensor,
        extras: torch.Tensor,
        *,
        available_actions: torch.Tensor | None = None,
        **kwargs,
    ):
        logits = self.forward(obs, extras, available_actions=available_actions, **kwargs)
        return torch.distributions.Categorical(logits=logits)

    def to_one_hot(self):
        class DiscreteOneHotActor(Actor[torch.distributions.OneHotCategorical]):
            def __init__(self, actor: CategoricalActor):
                super().__init__(
                    self.n_actions,
                    self.obs_shape,
                    self.extras_shape,
                    independent=self.independent,
                    n_agents=self.n_agents,
                )
                self.actor = actor

            def __hash__(self):
                return hash(self.name)

            def policy(
                self,
                obs: torch.Tensor,
                extras: torch.Tensor,
                *,
                available_actions: torch.Tensor | None = None,
                **kwargs,
            ):
                logits = self.actor.forward(obs, extras, available_actions=available_actions, **kwargs)
                return torch.distributions.OneHotCategorical(logits=logits)

        return DiscreteOneHotActor(self)


ContinuousDistribution = torch.distributions.Normal | torch.distributions.MultivariateNormal
ContinuousActor2 = Actor[ContinuousDistribution]


@dataclass
class ContinuousActor[T: ContinuousDistribution](Actor[torch.distributions.TransformedDistribution]):
    action_space: ContinuousSpace

    def __post_init__(self):
        super().__post_init__()
        self.transforms = [
            transforms.SigmoidTransform(),
            transforms.AffineTransform(
                loc=torch.tensor(self.action_space.low, device=self.device),
                scale=torch.tensor(self.action_space.high - self.action_space.low, device=self.device),
            ),
        ]

    def policy(
        self,
        obs: torch.Tensor,
        extras: torch.Tensor,
        *,
        available_actions: torch.Tensor | None = None,
        **kwargs,
    ):
        logits = self.forward(obs, extras, **kwargs)
        dist = self.make_distribution(logits)
        return torch.distributions.TransformedDistribution(dist, self.transforms)

    @abstractmethod
    def make_distribution(self, logits: torch.Tensor) -> T:
        """
        Create a distribution from the logits output by the NN.
        """

    @classmethod
    def from_env(cls, env: EnvConfig | MARLEnv, *, independent: bool = True, **kwargs):
        aspace = env.action_space
        assert isinstance(aspace, ContinuousSpace), "ContinuousActor can only be used with continuous action spaces."
        return super().from_env(env, independent=independent, action_space=aspace, **kwargs)


@dataclass
class NormalActor(ContinuousActor[torch.distributions.Normal]):
    def __post_init__(self):
        self.output_shape = (2 * self.n_actions,)
        return super().__post_init__()

    def make_distribution(self, logits: torch.Tensor):
        *dims, _ = logits.shape
        means = logits[..., : self.n_actions]
        stds = torch.nn.functional.softplus(logits[..., self.n_actions :])
        means = means.reshape(*dims, self.n_actions)
        stds = stds.reshape(*dims, self.n_actions)
        return torch.distributions.Normal(means, stds)


@dataclass
class MVNActor(ContinuousActor[torch.distributions.MultivariateNormal]):
    def __post_init__(self):
        # ,_actions for the means, n_actions ** 2 for the covariance matrix
        self.output_shape = (self.n_actions + self.n_actions**2,)
        self._eye = torch.eye(self.n_actions, device=self.device)
        return super().__post_init__()

    def make_distribution(self, logits: torch.Tensor):
        *dims, _ = logits.shape
        logits = logits.view(-1, self.output_size)
        means = logits[:, : self.n_actions]
        # Generate a Positive Definite covariance matrix
        # https://stackoverflow.com/questions/58176501/how-do-you-generate-positive-definite-matrix-in-pytorch
        raw_stds = logits[:, self.n_actions :].reshape(-1, self.n_actions, self.n_actions)
        positive_semi_definite = raw_stds @ raw_stds.transpose(1, 2)
        positive_definite = positive_semi_definite + self._eye
        # Shape back to the original shape
        means = means.reshape(*dims, self.n_actions)
        covariance_matrix = positive_definite.reshape(*dims, self.n_actions, self.n_actions)
        return torch.distributions.MultivariateNormal(means, covariance_matrix)
