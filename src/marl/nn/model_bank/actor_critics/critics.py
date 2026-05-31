from dataclasses import KW_ONLY, dataclass
from typing import Any, Sequence, cast

import torch
from marlenv import MARLEnv

from marl.env import EnvConfig
from marl.models.nn import (
    ActivationType,
    Critic,
    RecurrentNN,
)

from ..generic import CNN, MLP, RNN


def from_env(
    env: MARLEnv | EnvConfig,
    mlp_sizes: Sequence[int] = (256, 128),
    activation: ActivationType = "relu",
    independent: bool = True,
    recurrent: bool = False,
) -> Critic:
    registry: dict[tuple[int, bool, bool], type[Critic]] = {
        # (obs shape rank, discrete action space, recurrent)
        (3, True, False): (ConvCritic),
        (1, True, False): (LinearCritic),
        (3, True, True): (RecurrentCritic),
        (1, True, True): (RecurrentConvCritic),
    }
    config = (len(env.observation_shape), env.action_space.is_discrete, recurrent)
    network_class = registry.get(config)
    if network_class is not None:
        return cast(Any, network_class).from_env(env, mlp_sizes=mlp_sizes, activation=activation, independent=independent)
    err_msg = "\n".join(
        [
            f" - Shape Len: {shape_len}, Discrete: {is_discrete}, Recurrent: {is_recurrent}"
            for shape_len, is_discrete, is_recurrent in registry.keys()
        ]
    )
    raise NotImplementedError(f"Unsupported configuration: {config}.\nSupported combinations are:\n{err_msg}")


@dataclass
class ConvCritic(Critic):
    _: KW_ONLY
    kernel_sizes: Sequence[int] = (3, 3, 3)
    strides: Sequence[int] = (1, 1, 1)
    filters: Sequence[int] = (32, 64, 64)
    activation: ActivationType = "relu"
    mlp_sizes: Sequence[int] = (256, 128)
    independent: bool = False

    def __post_init__(self):
        super().__post_init__()
        assert len(self.obs_shape) == 3, "CNN can only handle 3D input shapes"
        self.cnn = CNN(
            self.obs_shape,
            hidden_activation=self.activation,
            output_activation=None,
            kernel_sizes=self.kernel_sizes,
            strides=self.strides,
            filters=self.filters,
        )
        self.mlp = MLP(
            (1,),
            self.cnn.output_size,
            self.extras_size,
            hidden_sizes=self.mlp_sizes,
            hidden_activation=self.activation,
            independent=self.independent,
            n_agents=self.n_agents,
        )

    def forward(self, obs: torch.Tensor, extras: torch.Tensor):
        x = self.cnn.forward(obs)
        return self.mlp.forward(x, extras)

    def __hash__(self):
        return id(self)


@dataclass
class LinearCritic(Critic):
    _: KW_ONLY
    activation: ActivationType = "relu"
    mlp_sizes: Sequence[int] = (256, 128)
    independent: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.mlp = MLP(
            (1,),
            self.obs_size,
            self.extras_size,
            hidden_sizes=self.mlp_sizes,
            hidden_activation=self.activation,
            independent=self.independent,
            n_agents=self.n_agents,
        )

    def forward(self, obs: torch.Tensor, extras: torch.Tensor):
        return self.mlp.forward(obs, extras)

    def __hash__(self):
        return id(self)


@dataclass
class RecurrentCritic(Critic):
    _: KW_ONLY
    activation: ActivationType = "relu"
    mlp_sizes: Sequence[int] = (256, 128)

    def __post_init__(self):
        super().__post_init__()
        self.rnn = RNN(
            (1,),
            self.obs_size,
            self.extras_size,
            hidden_activation=self.activation,
        )

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, *, masks: torch.Tensor | None = None, **kwargs):
        return self.rnn.forward(obs, extras, masks=masks, **kwargs)

    def reset_hidden_states(self):
        self.rnn.reset_hidden_states()

    def __hash__(self):
        return id(self)


@dataclass
class RecurrentConvCritic(Critic, RecurrentNN):
    _: KW_ONLY
    kernel_sizes: Sequence[int] = (3, 3, 3)
    strides: Sequence[int] = (1, 1, 1)
    filters: Sequence[int] = (32, 64, 64)
    activation: ActivationType = "relu"
    mlp_sizes: Sequence[int] = (256, 128)

    def __post_init__(self):
        super().__post_init__()
        assert len(self.obs_shape) == 3, "CNN can only handle 3D input shapes"
        self.cnn = CNN(
            self.obs_shape,
            hidden_activation=self.activation,
            output_activation=None,
            kernel_sizes=self.kernel_sizes,
            strides=self.strides,
            filters=self.filters,
        )
        self.rnn = RNN(
            (1,),
            self.cnn.output_size,
            self.extras_size,
            hidden_activation=self.activation,
        )

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, *, masks: torch.Tensor | None, **kwargs):
        x = self.cnn.forward(obs)
        return self.rnn.forward(x, extras, masks=masks, **kwargs)

    def reset_hidden_states(self):
        self.rnn.reset_hidden_states()

    def __hash__(self):
        return id(self)
