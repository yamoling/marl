from dataclasses import KW_ONLY, dataclass
from typing import Literal, Sequence, overload

import torch
from marlenv import ContinuousMARLEnv

from marl.env import EnvConfig
from marl.models.nn import ActivationType, ContinuousActor, MVNActor, NormalActor, RecurrentNN
from marl.models.nn.actor_critic import ContinuousDistribution

from ..generic import CNN, MLP, RNN


@overload
def from_env(
    env: ContinuousMARLEnv | EnvConfig[ContinuousMARLEnv],
    dist: Literal["normal"] = "normal",
    *,
    independent: bool = True,
    recurrent: bool = False,
    **init_kwargs,
) -> NormalActor: ...


@overload
def from_env(
    env: ContinuousMARLEnv | EnvConfig[ContinuousMARLEnv],
    dist: Literal["multivariate-normal"],
    *,
    independent: bool = True,
    recurrent: bool = False,
    **init_kwargs,
) -> MVNActor: ...


def from_env(
    env: ContinuousMARLEnv | EnvConfig[ContinuousMARLEnv],
    dist: Literal["normal", "multivariate-normal"] = "normal",
    *,
    independent: bool = True,
    recurrent: bool = False,
    **init_kwargs,
):
    registry: dict[
        tuple[int, Literal["normal", "multivariate-normal"], bool], type[ContinuousActor[ContinuousDistribution]]
    ] = {
        # (obs shape rank, discrete action space, recurrent)
        (1, "normal", False): NormalLinearActor,
        (1, "normal", True): NormalRecurrentActor,
        (3, "normal", False): NormalConvActor,
        (3, "normal", True): NormalRecurrentConvActor,
        (1, "multivariate-normal", False): MVNLinearActor,
    }
    config = (len(env.observation_shape), dist, recurrent)
    network_class = registry.get(config)
    if network_class is not None:
        return network_class.from_env(env, independent=independent, **init_kwargs)
    err_msg = "\n".join(
        [
            f" - Shape Len: {shape_len}, distribution: {dist}, recurrent: {is_recurrent}"
            for shape_len, dist, is_recurrent in registry.keys()
        ]
    )
    raise NotImplementedError(f"Unsupported configuration: {config}.\nSupported combinations are:\n{err_msg}")


@dataclass
class NormalConvActor(NormalActor):
    _: KW_ONLY
    kernel_sizes: Sequence[int] = (3, 3, 3)
    strides: Sequence[int] = (1, 1, 1)
    filters: Sequence[int] = (32, 64, 64)
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
            hidden_sizes=(256, 128),
            hidden_activation=self.activation,
            output_activation=None,
            independent=self.independent,
            n_agents=self.n_agents,
        )

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, **kwargs):
        return self.mlp.forward(obs, extras, **kwargs)

    def __hash__(self):
        return id(self)


@dataclass
class NormalLinearActor(NormalActor):
    _: KW_ONLY
    hidden_sizes: Sequence[int] = (256, 128)
    activation: ActivationType = "relu"

    def __post_init__(self):
        super().__post_init__()
        self.mlp = MLP(
            self.output_shape,
            self.obs_size,
            self.extras_size,
            hidden_sizes=self.hidden_sizes,
            hidden_activation=self.activation,
            output_activation=None,
            independent=self.independent,
            n_agents=self.n_agents,
        )

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, **kwargs):
        return self.mlp.forward(obs, extras, **kwargs)

    def __hash__(self):
        return id(self)


@dataclass
class NormalRecurrentActor(NormalActor, RecurrentNN):
    _: KW_ONLY
    mlp_head_sizes: Sequence[int] = (256,)
    mlp_tail_sizes: Sequence[int] = (128,)
    activation: ActivationType = "relu"
    n_grus: int = 1

    def __post_init__(self):
        super().__post_init__()
        self.rnn = RNN(
            self.output_shape,
            self.obs_size,
            self.extras_size,
            hidden_activation=self.activation,
            mlp_head_sizes=self.mlp_head_sizes,
            mlp_tail_sizes=self.mlp_tail_sizes,
            n_grus=self.n_grus,
        )

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, *, masks: torch.Tensor | None = None, **kwargs):
        return self.rnn.forward(obs, extras, masks=masks, **kwargs)

    def reset_hidden_states(self):
        return self.rnn.reset_hidden_states()

    def __hash__(self):
        return id(self)


@dataclass
class NormalRecurrentConvActor(NormalActor, RecurrentNN):
    _: KW_ONLY
    kernel_sizes: Sequence[int] = (3, 3, 3)
    strides: Sequence[int] = (1, 1, 1)
    filters: Sequence[int] = (32, 64, 64)
    mlp_head_sizes: Sequence[int] = (256,)
    mlp_tail_sizes: Sequence[int] = (128,)
    activation: ActivationType = "relu"
    n_grus: int = 1

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
        self.rnn = RNN(
            self.output_shape,
            self.cnn.output_size,
            self.extras_size,
            hidden_activation=self.activation,
            mlp_head_sizes=self.mlp_head_sizes,
            mlp_tail_sizes=self.mlp_tail_sizes,
            n_grus=self.n_grus,
        )

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, *, masks: torch.Tensor | None = None, **kwargs):
        cnn_out = self.cnn.forward(obs)
        return self.rnn.forward(cnn_out, extras, masks=masks, **kwargs)

    def reset_hidden_states(self):
        return self.rnn.reset_hidden_states()

    def __hash__(self):
        return id(self)


@dataclass
class MVNLinearActor(MVNActor):
    _: KW_ONLY
    hidden_sizes: Sequence[int] = (256, 128)
    activation: ActivationType = "relu"

    def __post_init__(self):
        super().__post_init__()
        self.mlp = MLP(
            self.output_shape,
            self.obs_size,
            self.extras_size,
            hidden_sizes=self.hidden_sizes,
            hidden_activation=self.activation,
            output_activation=None,
            independent=self.independent,
            n_agents=self.n_agents,
        )

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, **kwargs):
        return self.mlp.forward(obs, extras, **kwargs)

    def __hash__(self):
        return id(self)
