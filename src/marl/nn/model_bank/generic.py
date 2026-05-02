import math
from dataclasses import KW_ONLY, dataclass
from typing import Literal, Sequence, override

import torch
from marlenv import MARLEnv, MultiDiscreteSpace

from marl.models.nn import NN, RecurrentNN, get_activation

from ..layers import NoisyLinear
from ..utils import make_cnn


@dataclass(unsafe_hash=True)
class MLP(NN):
    """
    Multi layer perceptron
    """

    obs_size: int
    extras_size: int
    hidden_sizes: Sequence[int] = (256, 256)
    _: KW_ONLY
    noisy: bool = False
    output_activation: None | Literal["sigmoid", "tanh", "relu"] = None
    hidden_activation: Literal["sigmoid", "tanh", "relu"] = "relu"

    def __post_init__(self):
        NN.__post_init__(self)
        self.nn = torch.nn.Sequential()
        # [torch.nn.Linear(self.input_size, self.hidden_sizes[0]), torch.nn.ReLU()]
        for i in range(len(self.layer_sizes) - 1):
            self.nn.append(torch.nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
            self.nn.append(get_activation(self.hidden_activation))
        if self.noisy:
            self.nn.append(NoisyLinear(self.layer_sizes[-1], self.output_size))
        else:
            self.nn.append(torch.nn.Linear(self.layer_sizes[-1], self.output_size))
        if self.output_activation is not None:
            self.nn.append((get_activation(self.output_activation)))

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


@dataclass(unsafe_hash=True)
class CNN(NN):
    """
    CNN with three convolutional layers. The CNN output (output_cnn) is flattened and the extras are
    concatenated to this output. The CNN is followed by three linear layers (512, 256, output_shape[0]).
    """

    input_shape: tuple[int, int, int]
    extras_size: int
    mlp_sizes: Sequence[int] = (64, 64)
    mlp_noisy: bool = False
    hidden_activation: Literal["sigmoid", "tanh", "relu"] = "relu"
    output_activation: None | Literal["sigmoid", "tanh", "relu"] = None

    def __post_init__(self):
        super().__post_init__()
        kernel_sizes = [3, 3, 3]
        strides = [1, 1, 1]
        filters = [32, 64, 64]
        self.cnn, n_features = make_cnn(self.input_shape, filters, kernel_sizes, strides)
        self.linear = MLP(
            self.output_shape,
            n_features,
            self.extras_size,
            self.mlp_sizes,
            noisy=self.mlp_noisy,
            output_activation=self.output_activation,
        )

    @classmethod
    def qnetwork(cls, env: MARLEnv[MultiDiscreteSpace], mlp_sizes: tuple[int, ...] = (64, 64)):
        if env.reward_space.size == 1:
            output_shape = (env.n_actions,)
        else:
            output_shape = (env.n_actions, env.reward_space.size)
        assert len(env.observation_shape) == 3
        return cls(env.observation_shape, env.extras_shape[0], output_shape, mlp_sizes)  # type: ignore

    @override
    def forward(self, obs: torch.Tensor, extras: torch.Tensor, /, **kwargs) -> torch.Tensor:
        # For transitions, the shape is (batch_size, n_agents, channels, height, width)
        # For episodes, the shape is (time, batch_size, n_agents, channels, height, width)
        *dims, channels, height, width = obs.shape
        bs = math.prod(dims)
        obs = obs.reshape(bs, channels, height, width)
        features = self.cnn.forward(obs)
        extras = extras.reshape(bs, self.extras_size)
        res = self.linear.forward(features, extras)
        return res.view(*dims, *self.output_shape)


@dataclass(unsafe_hash=True)
class SimpleRNN(RecurrentNN):
    n_inputs: int
    hidden_size: int = 256

    def __post_init__(self):
        super().__post_init__()
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(self.n_inputs, self.hidden_size),
            torch.nn.ReLU(),
        )
        self.gru = torch.nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=False, num_layers=2)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.output_size)

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, /, masks: torch.Tensor | None = None, **kwargs):
        self.gru.flatten_parameters()
        assert len(obs.shape) >= 3, "The observation should have at most shape (ep_length, batch_size, obs_size)"
        # During batch training, the input has shape (episodes_length, batch_size, n_agents, obs_size).
        # This shape is not supported by the GRU layer, so we merge the batch_size and n_agents dimensions
        # while keeping the episode_length dimension.
        episode_length, *batch_agents, obs_size = obs.shape
        obs = obs.reshape(episode_length, -1, obs_size)
        extras = torch.reshape(extras, (*obs.shape[:-1], -1))
        x = torch.concat((obs, extras), dim=-1)
        x = self.fc1.forward(x)
        if masks is not None:
            episodes_lengths = masks.sum(0).cpu()
            x = torch.nn.utils.rnn.pack_padded_sequence(x, episodes_lengths, enforce_sorted=False)
            x, self.hidden_states = self.gru.forward(x, self.hidden_states)
            x, _ = torch.nn.utils.rnn.pad_packed_sequence(x)
        else:
            x, self.hidden_states = self.gru.forward(x, self.hidden_states)
        x = self.fc2.forward(x)
        # Restore the original shape of the batch
        x = x.view(episode_length, *batch_agents, self.output_size)
        return x
