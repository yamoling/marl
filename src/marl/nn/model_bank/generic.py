import math
from dataclasses import KW_ONLY, dataclass, field
from typing import Sequence, override

import torch

from marl.models.nn import NN, ActivationType, RecurrentNN, get_activation

from ..layers import ILinear
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
    independent: bool = False
    n_agents: int = -1

    def __post_init__(self):
        super().__post_init__()
        if self.independent:
            assert self.n_agents > 0, "n_agents must be greater than 0 when independent is True."
        self.nn = torch.nn.Sequential()
        for i in range(len(self.layer_sizes) - 1):
            in_size, out_size = self.layer_sizes[i], self.layer_sizes[i + 1]
            if self.independent:
                # Input layer has to transpose the input to be agent-wise and the output layer has to transpose the output back to the original shape.
                transpose_in = i == 0
                transpose_out = i == len(self.layer_sizes) - 2
                layer = ILinear(self.n_agents, in_size, out_size, transpose_in, transpose_out)
            else:
                layer = torch.nn.Linear(in_size, out_size)
            self.nn.append(layer)
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

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, /) -> torch.Tensor:
        *dims, _ = obs.shape
        obs = torch.concat((obs, extras), dim=-1)
        x = self.nn.forward(obs)
        return x.view(*dims, *self.output_shape)

    def __hash__(self):
        return id(self)


@dataclass
class CNN(NN):
    """
    CNN with three convolutional layers and a flattened output.
    """

    input_shape: tuple[int, int, int]
    _: KW_ONLY
    hidden_activation: ActivationType = "relu"
    output_activation: None | ActivationType = None
    kernel_sizes: Sequence[int] = (3, 3, 3)
    strides: Sequence[int] = (1, 1, 1)
    filters: Sequence[int] = (32, 64, 64)
    output_shape: tuple[int, ...] = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        assert len(self.strides) == len(self.kernel_sizes) == len(self.filters), (
            "The number of strides, kernel sizes and filters must be the same."
        )
        self.cnn, n_features = make_cnn(
            self.input_shape, self.filters, self.kernel_sizes, self.strides, self.hidden_activation
        )
        self.output_shape = (n_features,)

    def __hash__(self):
        return id(self)

    @override
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # For transitions, the shape is (batch_size, n_agents, channels, height, width)
        # For episodes, the shape is (time, batch_size, n_agents, channels, height, width)
        *dims, channels, height, width = obs.shape
        bs = math.prod(dims)
        obs = obs.reshape(bs, channels, height, width)
        x = self.cnn.forward(obs)
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
    independent: bool = False

    def __post_init__(self):
        if self.independent:
            raise NotImplementedError("Independent RNN is not implemented yet. Feel free to implement it !")
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

    def forward(
        self, obs: torch.Tensor, extras: torch.Tensor, *, masks: torch.Tensor | None = None, **kwargs
    ) -> torch.Tensor:
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
