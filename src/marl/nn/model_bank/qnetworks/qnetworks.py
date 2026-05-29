from dataclasses import KW_ONLY, dataclass
from typing import Sequence

import torch
from marlenv import DiscreteMARLEnv

from marl.env import EnvConfig
from marl.models.nn import ActivationType, QNetwork, RecurrentQNetwork
from marl.nn.layers import NoisyLinear
from marl.nn.model_bank.generic import CNN, MLP, RNN
from marl.nn.utils import MySequential


@dataclass
class QCNN(QNetwork):
    _: KW_ONLY
    mlp_sizes: Sequence[int] = (256, 128)
    hidden_activation: ActivationType = "relu"

    def __post_init__(self):
        super().__post_init__()
        assert len(self.obs_shape) == 3
        self.cnn = CNN(self.obs_shape, hidden_activation=self.hidden_activation, output_activation=self.hidden_activation)
        if self.noisy:
            self.noisy_layer = NoisyLinear(self.mlp_sizes[-1], self.output_size)
            self.mlp = MySequential(
                MLP(
                    (self.mlp_sizes[-1],),
                    self.cnn.output_size,
                    self.extras_size,
                    hidden_sizes=self.mlp_sizes[:-1],
                    hidden_activation=self.hidden_activation,
                    output_activation=self.hidden_activation,
                    independent=self.independent,
                    n_agents=self.n_agents,
                ),
                NoisyLinear(self.mlp_sizes[-1], self.output_size),
            )
        else:
            self.mlp = MLP(
                self.output_shape,
                self.cnn.output_size,
                self.extras_size,
                hidden_sizes=self.mlp_sizes,
                independent=self.independent,
                n_agents=self.n_agents,
            )

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, /, **kwargs):
        x = self.cnn.forward(obs)
        x = self.mlp.forward(x, extras, **kwargs)
        return x

    def __hash__(self):
        return id(self)


@dataclass
class QMLP(QNetwork):
    _: KW_ONLY
    hidden_sizes: Sequence[int] = (256, 128)
    activation: ActivationType = "relu"
    noisy: bool = False

    def __post_init__(self):
        super().__post_init__()

        if self.noisy:
            self.nn = MySequential(
                MLP(
                    (self.hidden_sizes[-1],),
                    self.obs_size,
                    self.extras_size,
                    hidden_sizes=self.hidden_sizes,
                    hidden_activation=self.activation,
                    output_activation=self.activation,
                    independent=self.independent,
                    n_agents=self.n_agents,
                ),
                NoisyLinear(self.hidden_sizes[1], self.n_actions),
            )
        else:
            self.nn = MLP(
                self.output_shape,
                self.obs_size,
                self.extras_size,
                hidden_sizes=self.hidden_sizes,
                hidden_activation=self.activation,
                independent=self.independent,
                n_agents=self.n_agents,
            )

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, /, **kwargs):
        return self.nn.forward(obs, extras, **kwargs)

    def __hash__(self):
        return id(self)

    @classmethod
    def from_env(
        cls,
        env: EnvConfig[DiscreteMARLEnv] | DiscreteMARLEnv,
        hidden_sizes: Sequence[int] = (256, 128),
        activation: ActivationType = "relu",
        noisy: bool = False,
        duelling: bool = False,
        **kwargs,
    ):
        return super().from_env(env, activation=activation, hidden_size=hidden_sizes, noisy=noisy, **kwargs)


@dataclass
class QRNN(RecurrentQNetwork):
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

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, /, **kwargs) -> torch.Tensor:
        return self.rnn.forward(obs, extras, **kwargs)

    def reset_hidden_states(self):
        return self.rnn.reset_hidden_states()

    def __hash__(self):
        return id(self)


@dataclass
class QCRNN(RecurrentQNetwork):
    _: KW_ONLY
    mlp_head_sizes: Sequence[int] = (256,)
    mlp_tail_sizes: Sequence[int] = (128,)
    n_grus: int = 1
    hidden_activation: ActivationType = "relu"
    kernel_sizes: Sequence[int] = (3, 3, 3)
    strides: Sequence[int] = (1, 1, 1)
    filters: Sequence[int] = (32, 64, 64)

    def __post_init__(self):
        super().__post_init__()
        assert len(self.obs_shape) == 3
        self.cnn = CNN(self.obs_shape, hidden_activation=self.hidden_activation, output_activation=self.hidden_activation)
        self.rnn = RNN(
            self.output_shape,
            self.cnn.output_size,
            self.extras_size,
            mlp_head_sizes=self.mlp_head_sizes,
            mlp_tail_sizes=self.mlp_tail_sizes,
            n_grus=self.n_grus,
            hidden_activation=self.hidden_activation,
        )

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, /, **kwargs):
        x = self.cnn.forward(obs)
        x = self.rnn.forward(x, extras, **kwargs)
        return x

    def __hash__(self):
        return id(self)

    def reset_hidden_states(self):
        return self.rnn.reset_hidden_states()
