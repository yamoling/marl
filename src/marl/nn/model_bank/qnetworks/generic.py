from dataclasses import KW_ONLY, dataclass
from typing import Sequence

import torch

from marl.models.nn import ActivationType, QNetwork, RecurrentQNetwork
from marl.nn.layers import NoisyLinear
from marl.nn.model_bank.generic import CNN, CRNN, MLP, RNN


@dataclass
class QCNN(QNetwork):
    _: KW_ONLY
    mlp_sizes: Sequence[int] = (256, 128)
    hidden_activation: ActivationType = "relu"
    noisy: bool = False

    def __post_init__(self):
        super().__post_init__()
        assert len(self.obs_shape) == 3
        if self.noisy:
            self.noisy_layer = NoisyLinear(self.mlp_sizes[-1], self.output_size)
            cnn_output_shape = (self.mlp_sizes[-1],)
            mlp_sizes = self.mlp_sizes[:-1]
            cnn_output_activation = self.hidden_activation
        else:
            self.noisy_layer = None
            cnn_output_shape = self.output_shape
            cnn_output_activation = None
            mlp_sizes = self.mlp_sizes
        self.cnn = CNN(
            cnn_output_shape,
            self.obs_shape,
            self.extras_size,
            mlp_sizes=mlp_sizes,
            hidden_activation=self.hidden_activation,
            output_activation=cnn_output_activation,
        )

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, /, **kwargs):
        x = self.cnn.forward(obs, extras, **kwargs)
        if self.noisy_layer is not None:
            return self.noisy_layer.forward(x)
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
            self.nn = MLP(
                (self.hidden_sizes[-1],),
                self.obs_size,
                self.extras_size,
                hidden_sizes=self.hidden_sizes,
                hidden_activation=self.activation,
                output_activation=self.activation,
            )
            self.noisy_layer = NoisyLinear(self.hidden_sizes[1], self.n_actions)
        else:
            self.nn = MLP(
                self.output_shape, self.obs_size, self.extras_size, hidden_sizes=self.hidden_sizes, hidden_activation=self.activation
            )
            self.noisy_layer = None

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, /, **kwargs):
        x = self.nn.forward(obs, extras, **kwargs)
        if self.noisy_layer is not None:
            return self.noisy_layer.forward(x)
        return x

    def __hash__(self):
        return id(self)


@dataclass
class QRNN(RecurrentQNetwork):
    _: KW_ONLY
    hidden_sizes: Sequence[int] = (256, 128)
    activation: ActivationType = "relu"

    def __post_init__(self):
        super().__post_init__()
        self.rnn = RNN(self.output_shape, self.obs_size, self.extras_size, self.hidden_sizes, self.activation)

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, /, **kwargs) -> torch.Tensor:
        return self.rnn.forward(obs, extras, **kwargs)

    def reset_hidden_states(self):
        return self.rnn.reset_hidden_states()

    def __hash__(self):
        return id(self)


@dataclass
class QCRNN(RecurrentQNetwork):
    _: KW_ONLY
    mlp_sizes: Sequence[int] = (256, 128)
    hidden_activation: ActivationType = "relu"
    kernel_sizes: Sequence[int] = (3, 3, 3)
    strides: Sequence[int] = (1, 1, 1)
    filters: Sequence[int] = (32, 64, 64)

    def __post_init__(self):
        super().__post_init__()
        assert len(self.obs_shape) == 3
        self.nn = CRNN(
            self.output_shape,
            self.obs_shape,
            self.extras_size,
            self.mlp_sizes,
            self.hidden_activation,
            kernel_sizes=self.kernel_sizes,
            strides=self.strides,
            filters=self.filters,
        )

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, /, **kwargs) -> torch.Tensor:
        return self.nn.forward(obs, extras, **kwargs)

    def __hash__(self):
        return id(self)
