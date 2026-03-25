import math
from dataclasses import dataclass, field
from typing import Sequence

import torch
from marlenv import MARLEnv, MultiDiscreteSpace

from marl.models import NN, RecurrentNN

from ..layers import NoisyLinear
from ..utils import make_cnn


@dataclass(unsafe_hash=True)
class MLP(NN):
    """
    Multi layer perceptron
    """

    layer_sizes: Sequence[int] = field(init=False)

    def __init__(
        self,
        input_size: int,
        extras_size: int,
        hidden_sizes: Sequence[int],
        output_shape: tuple[int, ...],
        last_layer_noisy: bool = False,
    ):
        super().__init__(output_shape)
        self.output_shape = output_shape
        output_size = math.prod(self.output_shape)
        self.layer_sizes = (input_size + extras_size, *hidden_sizes, output_size)
        layers = [torch.nn.Linear(input_size + extras_size, hidden_sizes[0]), torch.nn.ReLU()]
        for i in range(len(hidden_sizes) - 1):
            layers.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(torch.nn.ReLU())
        if last_layer_noisy:
            layers.append(NoisyLinear(hidden_sizes[-1], output_size))
        else:
            layers.append(torch.nn.Linear(hidden_sizes[-1], output_size))
        self.nn = torch.nn.Sequential(*layers)

    @classmethod
    def qnetwork(
        cls,
        env: MARLEnv[MultiDiscreteSpace],
        hidden_sizes: Sequence[int] = (64,),
        last_layer_noisy: bool = False,
    ):
        if env.is_multi_objective:
            output_shape = (env.n_actions, env.reward_space.size)
        else:
            output_shape = (env.n_actions,)
        return cls(
            env.observation_shape[0],
            env.extras_shape[0],
            tuple(hidden_sizes),
            output_shape,
            last_layer_noisy,
        )

    def forward(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        *dims, _obs_size = obs.shape
        obs = torch.concat((obs, extras), dim=-1)
        x = self.nn(obs)
        return x.view(*dims, *self.output_shape)


@dataclass(unsafe_hash=True)
class CNN(NN):
    """
    CNN with three convolutional layers. The CNN output (output_cnn) is flattened and the extras are
    concatenated to this output. The CNN is followed by three linear layers (512, 256, output_shape[0]).
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        extras_size: int,
        output_shape: int | tuple[int, ...],
        mlp_sizes: tuple[int, ...] = (64, 64),
        mlp_noisy: bool = False,
    ):
        assert len(input_shape) == 3, f"CNN can only handle 3D input shapes ({len(input_shape)} here)"
        super().__init__(output_shape)
        self.extras_size = extras_size
        kernel_sizes = [3, 3, 3]
        strides = [1, 1, 1]
        filters = [32, 64, 64]
        self.cnn, n_features = make_cnn(input_shape, filters, kernel_sizes, strides)
        if isinstance(output_shape, int):
            output_shape = (output_shape,)
        self.output_shape = output_shape
        self.linear = MLP(n_features, extras_size, mlp_sizes, self.output_shape, last_layer_noisy=mlp_noisy)

    @classmethod
    def qnetwork(cls, env: MARLEnv[MultiDiscreteSpace], mlp_sizes: tuple[int, ...] = (64, 64)):
        if env.reward_space.size == 1:
            output_shape = (env.n_actions,)
        else:
            output_shape = (env.n_actions, env.reward_space.size)
        assert len(env.observation_shape) == 3
        return cls(env.observation_shape, env.extras_shape[0], output_shape, mlp_sizes)

    def forward(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        # For transitions, the shape is (batch_size, n_agents, channels, height, width)
        # For episodes, the shape is (time, batch_size, n_agents, channels, height, width)
        *dims, channels, height, width = obs.shape
        bs = math.prod(dims)
        # obs = obs.view(bs, channels, height, width)
        obs = obs.reshape(bs, channels, height, width)
        features = self.cnn.forward(obs)
        # extras = extras.view(bs, *self.extras_shape)
        extras = extras.reshape(bs, self.extras_size)
        res = self.linear.forward(features, extras)
        return res.view(*dims, *self.output_shape)


@dataclass(unsafe_hash=True)
class SimpleRNN(RecurrentNN):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        hidden_size: int = 256,
    ):
        super().__init__(n_outputs)
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(n_inputs, hidden_size),
            torch.nn.ReLU(),
        )
        self.gru = torch.nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=False, num_layers=2)
        self.fc2 = torch.nn.Linear(hidden_size, n_outputs)
        self.n_outputs = n_outputs

    def forward(self, obs: torch.Tensor, extras: torch.Tensor):
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
        x, self.hidden_states = self.gru.forward(x, self.hidden_states)
        x = self.fc2.forward(x)
        # Restore the original shape of the batch
        x = x.view(episode_length, *batch_agents, self.n_outputs)
        return x
