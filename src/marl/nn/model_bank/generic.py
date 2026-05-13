import math
from dataclasses import KW_ONLY, dataclass
from typing import Sequence, override

import torch

from marl.models.nn import NN, ActivationType, RecurrentNN, get_activation

from ..utils import make_cnn


@dataclass
class MLP(NN):
    """
    Multi layer perceptron
    """

    obs_size: int
    extras_size: int
    _: KW_ONLY
    hidden_sizes: Sequence[int] = (128, 256, 128)
    hidden_activation: ActivationType = "relu"
    output_activation: None | ActivationType = None

    def __post_init__(self):
        super().__post_init__()
        self.nn = torch.nn.Sequential()
        for i in range(len(self.layer_sizes) - 1):
            self.nn.append(torch.nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
            if i < len(self.layer_sizes) - 2 or self.output_activation is not None:
                self.nn.append(get_activation(self.hidden_activation))

    @property
    def output_size(self):
        return math.prod(self.output_shape)

    @property
    def input_size(self):
        return self.obs_size + self.extras_size

    @property
    def layer_sizes(self) -> tuple[int, ...]:
        return self.input_size, *self.hidden_sizes, self.output_size

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, /, **kwargs) -> torch.Tensor:
        *dims, _ = obs.shape
        obs = torch.concat((obs, extras), dim=-1)
        x = self.nn.forward(obs)
        return x.view(*dims, *self.output_shape)

    def __hash__(self):
        return id(self)


@dataclass
class CNN(NN):
    """
    CNN with three convolutional layers. The CNN output (output_cnn) is flattened and the extras are
    concatenated to this output. The CNN is followed by three linear layers of shape (*mlp_sizes, output_shape[0]).
    """

    input_shape: tuple[int, int, int]
    extras_size: int
    _: KW_ONLY
    mlp_sizes: Sequence[int] | None = (256, 128)
    hidden_activation: ActivationType = "relu"
    output_activation: None | ActivationType = None
    kernel_sizes: Sequence[int] = (3, 3, 3)
    strides: Sequence[int] = (1, 1, 1)
    filters: Sequence[int] = (32, 64, 64)

    def __post_init__(self):
        super().__post_init__()
        assert len(self.strides) == len(self.kernel_sizes) == len(self.filters), (
            "The number of strides, kernel sizes and filters must be the same."
        )
        self.cnn, n_features = make_cnn(self.input_shape, self.filters, self.kernel_sizes, self.strides, self.hidden_activation)
        self.linear = None
        if self.mlp_sizes is not None:
            self.linear = MLP(
                self.output_shape,
                n_features,
                self.extras_size,
                hidden_sizes=self.mlp_sizes,
                hidden_activation=self.hidden_activation,
                output_activation=self.output_activation,
            )
        else:
            self.output_shape = (n_features,)

    def __hash__(self):
        return id(self)

    @override
    def forward(self, obs: torch.Tensor, extras: torch.Tensor, /, **kwargs) -> torch.Tensor:
        # For transitions, the shape is (batch_size, n_agents, channels, height, width)
        # For episodes, the shape is (time, batch_size, n_agents, channels, height, width)
        *dims, channels, height, width = obs.shape
        bs = math.prod(dims)
        obs = obs.reshape(bs, channels, height, width)
        x = self.cnn.forward(obs)
        if self.linear is not None:
            extras = extras.reshape(bs, self.extras_size)
            x = self.linear.forward(x, extras)
        return x.view(*dims, *self.output_shape)


@dataclass
class RNN(RecurrentNN):
    obs_size: int
    extras_size: int
    _: KW_ONLY
    mlp_head_sizes: Sequence[int] = (256,)
    mlp_tail_sizes: Sequence[int] = (128,)
    n_grus: int = 1
    hidden_activation: ActivationType = "relu"

    def __post_init__(self):
        super().__post_init__()
        self.head = torch.nn.Sequential()
        self.tail = torch.nn.Sequential()
        for dim, next_dim in zip(self.head_layer_sizes[:-1], self.head_layer_sizes[1:]):
            self.head.append(torch.nn.Linear(dim, next_dim))
            self.head.append(get_activation(self.hidden_activation))
        self.gru = torch.nn.GRU(self.mlp_head_sizes[-1], self.mlp_tail_sizes[0], batch_first=False)
        for dim, next_dim in zip(self.tail_layer_sizes[:-1], self.tail_layer_sizes[1:]):
            self.tail.append(torch.nn.Linear(dim, next_dim))
            self.tail.append(get_activation(self.hidden_activation))
        # Remove the last layer activation
        self.tail.pop(-1)

    @property
    def input_size(self):
        return self.obs_size + self.extras_size

    @property
    def head_layer_sizes(self):
        return [self.input_size, *self.mlp_head_sizes]

    @property
    def tail_layer_sizes(self):
        return [*self.mlp_tail_sizes, self.output_size]

    def __hash__(self):
        return id(self)

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, /, masks: torch.Tensor | None = None, **kwargs):
        self.gru.flatten_parameters()
        assert len(obs.shape) >= 3, "The observation should have at least shape (ep_length, batch_size, obs_size)"
        # During batch training, the input has shape (episodes_length, batch_size, n_agents, obs_size).
        # This shape is not supported by the GRU layer, so we merge the batch_size and n_agents dimensions
        # while keeping the episode_length dimension.
        episode_length, *batch_size, n_agents, obs_size = obs.shape
        obs = obs.reshape(episode_length, -1, obs_size)
        extras = torch.reshape(extras, (*obs.shape[:-1], self.extras_size))
        x = torch.concat((obs, extras), dim=-1)
        x = self.head.forward(x)
        if masks is not None:
            episodes_lengths = masks.long().sum(0).cpu()
            episodes_lengths = episodes_lengths.repeat_interleave(n_agents)
            packed = torch.nn.utils.rnn.pack_padded_sequence(x, episodes_lengths, enforce_sorted=False)
            packed, self._hidden_states = self.gru.forward(packed, self._hidden_states)
            x, _ = torch.nn.utils.rnn.pad_packed_sequence(packed)
        else:
            x, self._hidden_states = self.gru.forward(x, self._hidden_states)
        x = self.tail.forward(x)
        # Restore the original shape of the batch
        x = x.view(episode_length, *batch_size, n_agents, self.output_size)
        return x


@dataclass
class CRNN(RecurrentNN):
    """Convolutional Recurrent Neural Network."""

    input_shape: tuple[int, int, int]
    extras_size: int
    hidden_activation: ActivationType
    _: KW_ONLY
    mlp_head_sizes: Sequence[int] = (256,)
    mlp_tail_sizes: Sequence[int] = (128,)
    output_activation: None | ActivationType = None
    kernel_sizes: Sequence[int] = (3, 3, 3)
    strides: Sequence[int] = (1, 1, 1)
    filters: Sequence[int] = (32, 64, 64)

    def __post_init__(self):
        super().__post_init__()
        self.cnn = CNN((0,), self.input_shape, 0, mlp_sizes=None)
        # self.cnn, n_features = make_cnn(self.input_shape, self.filters, self.kernel_sizes, self.strides, self.hidden_activation)
        self.rnn = RNN(
            self.output_shape,
            self.cnn.output_size,
            self.extras_size,
            mlp_head_sizes=self.mlp_head_sizes,
            mlp_tail_sizes=self.mlp_tail_sizes,
            hidden_activation=self.hidden_activation,
        )

    def __hash__(self):
        return id(self)

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, *, masks: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        features = self.cnn.forward(obs, extras)
        return self.rnn.forward(features, extras, masks=masks, **kwargs)

    def reset_hidden_states(self):
        return self.rnn.reset_hidden_states()
