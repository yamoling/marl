from dataclasses import KW_ONLY, dataclass
from typing import Sequence

import torch
from marlenv import MARLEnv

from marl.env import EnvConfig
from marl.models.nn import ActivationType, CategoricalActor, RecurrentNN

from ..generic import CNN, MLP, RNN


def from_env(env: MARLEnv | EnvConfig, recurrent: bool, *, independent: bool = True, **init_kwargs) -> CategoricalActor:
    assert env.action_space.is_discrete, "Only discrete action spaces are supported by the discrete actor factory."
    registry = {
        (1, False): CategoricalLinearActor,
        (1, True): CategoricalRecurrentActor,
        (3, False): CategoricalConvActor,
        (3, True): CategoricalRecurrentConvActor,
    }
    config = (len(env.observation_shape), recurrent)
    network_class = registry.get(config)
    if network_class is not None:
        return network_class.from_env(env, independent=independent, **init_kwargs)
    err_msg = "\n".join(
        [f" - Shape Len: {shape_len}, Recurrent: {is_recurrent}" for shape_len, is_recurrent in registry.keys()]
    )
    raise NotImplementedError(f"Unsupported configuration: {config}.\nSupported combinations are:\n{err_msg}")


@dataclass
class CategoricalLinearActor(CategoricalActor):
    """Categorical Linear Actor"""

    _: KW_ONLY
    mlp_sizes: Sequence[int] = (256, 192, 128)
    activation: ActivationType = "relu"
    independent: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.mlp = MLP(
            (self.n_actions,),
            self.obs_size,
            self.extras_size,
            hidden_sizes=self.mlp_sizes,
            hidden_activation=self.activation,
            output_activation=None,
            independent=self.independent,
            n_agents=self.n_agents,
        )

    @classmethod
    def from_env(
        cls,
        env: MARLEnv | EnvConfig,
        independent: bool = False,
        mlp_sizes: Sequence[int] = (256, 128),
        activation: ActivationType = "relu",
        **kwargs,
    ):
        return super().from_env(env, mlp_sizes=mlp_sizes, activation=activation, independent=independent, **kwargs)

    def forward(
        self,
        obs: torch.Tensor,
        extras: torch.Tensor,
        *,
        available_actions: torch.Tensor | None = None,
        **kwargs,
    ):
        logits = self.mlp(obs, extras, **kwargs)
        return self.mask(logits, available_actions)

    def __hash__(self):
        return id(self)


class CategoricalRecurrentActor(CategoricalActor, RecurrentNN):
    """Categorical Recurrent Actor"""

    _: KW_ONLY
    mlp_head_sizes: Sequence[int] = (256,)
    mlp_tail_sizes: Sequence[int] = (128,)
    activation: ActivationType = "relu"

    def __post_init__(self):
        super().__post_init__()
        self.rnn = RNN(
            self.output_shape,
            self.obs_size,
            self.extras_size,
            mlp_head_sizes=self.mlp_head_sizes,
            mlp_tail_sizes=self.mlp_tail_sizes,
            hidden_activation=self.activation,
        )

    @classmethod
    def from_env(
        cls,
        env: MARLEnv | EnvConfig,
        independent: bool = False,
        mlp_head_sizes: Sequence[int] = (256,),
        mlp_tail_sizes: Sequence[int] = (128,),
        activation: ActivationType = "relu",
        **kwargs,
    ):
        return super().from_env(
            env,
            independent=independent,
            mlp_head_sizes=mlp_head_sizes,
            mlp_tail_sizes=mlp_tail_sizes,
            activation=activation,
        )

    def forward(
        self,
        obs: torch.Tensor,
        extras: torch.Tensor,
        *,
        available_actions: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
        **kwargs,
    ):
        logits = self.rnn.forward(obs, extras, masks=masks, **kwargs)
        return self.mask(logits, available_actions)

    def reset_hidden_states(self):
        return self.rnn.reset_hidden_states()

    def __hash__(self):
        return id(self)


@dataclass
class CategoricalConvActor(CategoricalActor):
    """Categorical convonultional actor network"""

    _: KW_ONLY
    kernel_sizes: Sequence[int] = (3, 3, 3)
    strides: Sequence[int] = (1, 1, 1)
    filters: Sequence[int] = (32, 64, 64)
    mlp_sizes: Sequence[int] = (256, 128)
    activation: ActivationType = "relu"

    def __post_init__(self):
        super().__post_init__()
        assert len(self.obs_shape) == 3
        self.cnn = CNN(
            self.obs_shape,
            hidden_activation=self.activation,
            output_activation=None,
            kernel_sizes=self.kernel_sizes,
            strides=self.strides,
            filters=self.filters,
        )
        self.mlp = MLP(
            (self.n_actions,),
            self.cnn.output_size,
            self.extras_size,
            hidden_sizes=self.mlp_sizes,
            hidden_activation=self.activation,
            output_activation=None,
            independent=self.independent,
            n_agents=self.n_agents,
        )

    def forward(
        self,
        obs: torch.Tensor,
        extras: torch.Tensor,
        *,
        available_actions: torch.Tensor | None = None,
        **kwargs,
    ):
        x = self.cnn.forward(obs)
        logits = self.mlp.forward(x, extras)
        return self.mask(logits, available_actions)

    def __hash__(self):
        return hash(self.name)


@dataclass
class CategoricalRecurrentConvActor(CategoricalConvActor, RecurrentNN):
    """Categorical Recurrent Convolutional Actor"""

    mlp_head_sizes: Sequence[int] = (256,)
    mlp_tail_sizes: Sequence[int] = (128,)

    def __post_init__(self):
        super().__post_init__()
        self.rnn = RNN(
            self.output_shape,
            self.cnn.output_size,
            self.extras_size,
            mlp_head_sizes=self.mlp_head_sizes,
            mlp_tail_sizes=self.mlp_tail_sizes,
            hidden_activation=self.activation,
        )

    def forward(
        self,
        obs: torch.Tensor,
        extras: torch.Tensor,
        *,
        available_actions: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
        **kwargs,
    ):
        x = self.cnn.forward(obs)
        logits = self.rnn.forward(x, extras, masks=masks, **kwargs)
        return self.mask(logits, available_actions, replacement=-torch.inf)

    def reset_hidden_states(self):
        return self.rnn.reset_hidden_states()

    def __hash__(self):
        return id(self)
