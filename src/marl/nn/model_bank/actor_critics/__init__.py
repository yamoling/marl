from typing import Literal, overload

from marlenv import ContinuousMARLEnv, DiscreteMARLEnv

from marl.env import EnvConfig
from marl.models.nn import CategoricalActor, Critic, MVNActor, NormalActor

from . import continuous_actors, critics, discrete_actors
from .continuous_actors import NormalConvActor, NormalLinearActor, NormalRecurrentActor, NormalRecurrentConvActor
from .critics import ConvCritic, LinearCritic, RecurrentConvCritic, RecurrentCritic
from .discrete_actors import (
    CategoricalConvActor,
    CategoricalLinearActor,
    CategoricalRecurrentActor,
    CategoricalRecurrentConvActor,
)


@overload
def from_env(
    env: EnvConfig[DiscreteMARLEnv],
    recurrent: bool,
    *,
    independent: bool = True,
    critic_kwargs: dict | None = None,
    actor_kwargs: dict | None = None,
) -> tuple[CategoricalActor, Critic]: ...


@overload
def from_env(
    env: EnvConfig[ContinuousMARLEnv],
    recurrent: bool,
    *,
    independent: bool = True,
    dist: Literal["normal"] = "normal",
    critic_kwargs: dict | None = None,
    actor_kwargs: dict | None = None,
) -> tuple[NormalActor, Critic]: ...


@overload
def from_env(
    env: EnvConfig[ContinuousMARLEnv],
    recurrent: bool,
    *,
    independent: bool = True,
    dist: Literal["multivariate-normal"],
    critic_kwargs: dict | None = None,
    actor_kwargs: dict | None = None,
) -> tuple[MVNActor, Critic]: ...


def from_env(
    env: EnvConfig,
    recurrent: bool,
    *,
    independent: bool = True,
    dist: Literal["normal", "multivariate-normal"] = "normal",
    critic_kwargs: dict | None = None,
    actor_kwargs: dict | None = None,
):
    critic_kwargs = critic_kwargs or {}
    actor_kwargs = actor_kwargs or {}
    critic = critics.from_env(env, independent=independent, recurrent=recurrent, **critic_kwargs)
    if env.action_space.is_discrete:
        actor = discrete_actors.from_env(env, independent=independent, recurrent=recurrent, **actor_kwargs)
    else:
        actor = continuous_actors.from_env(env, independent=independent, recurrent=recurrent, dist=dist, **actor_kwargs)
    return actor, critic


__all__ = [
    "from_env",
    "CategoricalConvActor",
    "CategoricalLinearActor",
    "CategoricalRecurrentActor",
    "CategoricalRecurrentConvActor",
    "NormalConvActor",
    "NormalLinearActor",
    "NormalRecurrentActor",
    "NormalRecurrentConvActor",
    "ConvCritic",
    "LinearCritic",
    "RecurrentConvCritic",
    "RecurrentCritic",
]
