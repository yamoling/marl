from typing import Optional, Iterable
from dataclasses import dataclass
from rlenv.models import RLEnv
import torch
import math

from marl.models.nn import QNetwork, RecurrentQNetwork, ActorCriticNN, NN


@dataclass(unsafe_hash=True)
class MLP(QNetwork):
    """
    Multi layer perceptron
    """

    layer_sizes: tuple[int, ...]

    def __init__(
        self,
        input_size: int,
        extras_size: int,
        hidden_sizes: tuple[int, ...],
        output_shape: tuple[int, ...],
    ):
        super().__init__((input_size,), (extras_size,), output_shape)
        self.layer_sizes = (input_size + extras_size, *hidden_sizes, math.prod(output_shape))
        layers = [torch.nn.Linear(input_size + extras_size, hidden_sizes[0]), torch.nn.ReLU()]
        for i in range(len(hidden_sizes) - 1):
            layers.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(hidden_sizes[-1], math.prod(output_shape)))
        self.nn = torch.nn.Sequential(*layers)

    @classmethod
    def from_env(cls, env: RLEnv, hidden_sizes: Optional[Iterable[int]] = None):
        if hidden_sizes is None:
            hidden_sizes = (64,)
        return MLP(
            env.observation_shape[0],
            env.extra_feature_shape[0],
            tuple(hidden_sizes),
            (env.n_actions, env.reward_size),
        )

    def forward(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        *dims, _obs_size = obs.shape
        obs = torch.concat((obs, extras), dim=-1)
        x = self.nn(obs)
        return x.view(*dims, *self.output_shape)


class RNNQMix(RecurrentQNetwork):
    """RNN used in the QMix paper:
    - linear 64
    - relu
    - GRU 64
    - relu
    - linear 64"""

    def __init__(self, input_shape: tuple[int, ...], extras_shape: tuple[int, ...], output_shape: tuple[int, ...]):
        assert len(input_shape) == 1, "RNNQMix can only handle 1D inputs"
        assert len(extras_shape) == 1, "RNNQMix can only handle 1D extras"
        super().__init__(input_shape, extras_shape, output_shape)
        n_inputs = input_shape[0] + extras_shape[0]
        n_outputs = math.prod(output_shape)
        self.fc1 = torch.nn.Sequential(torch.nn.Linear(n_inputs, 64), torch.nn.ReLU())
        self.gru = torch.nn.GRU(input_size=64, hidden_size=64, batch_first=False)
        self.fc2 = torch.nn.Linear(64, n_outputs)

    def forward(self, obs: torch.Tensor, extras: torch.Tensor):
        self.gru.flatten_parameters()
        assert len(obs.shape) >= 3, "The observation should have at least shape (ep_length, batch_size, obs_size)"
        # During batch training, the input has shape (episodes_length, batch_size, n_agents, obs_size).
        # This shape is not supported by the GRU layer, so we merge the batch_size and n_agents dimensions
        # while keeping the episode_length dimension.
        try:
            episode_length, *batch_agents, obs_size = obs.shape
            obs = obs.reshape(episode_length, -1, obs_size)
            extras = torch.reshape(extras, (*obs.shape[:-1], -1))
            x = torch.concat((obs, extras), dim=-1)
            # print(x)
            x = self.fc1.forward(x)
            x, self.hidden_states = self.gru.forward(x, self.hidden_states)
            x = self.fc2.forward(x)
            # Restore the original shape of the batch
            x = x.view(episode_length, *batch_agents, *self.output_shape)
            return x
        except RuntimeError as e:
            error_message = str(e)
            if "shape" in error_message:
                error_message += "\nDid you use a TransitionMemory instead of an EpisodeMemory alongside an RNN ?"
            raise RuntimeError(error_message)


RNN = RNNQMix


class DuelingMLP(QNetwork):
    def __init__(self, nn: QNetwork, output_size: int):
        assert len(nn.output_shape) == 1
        super().__init__(nn.input_shape, nn.extras_shape, (output_size,))
        self.nn = nn
        self.value_head = torch.nn.Linear(nn.output_shape[0], 1)
        self.advantage = torch.nn.Linear(nn.output_shape[0], output_size)

    @classmethod
    def from_env(cls, env: RLEnv, nn: QNetwork):
        assert nn.input_shape == env.observation_shape
        assert nn.extras_shape == env.extra_feature_shape
        return cls(nn, env.n_actions)

    def forward(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        features = self.nn.forward(obs, extras)
        features = torch.nn.functional.relu(features)
        value = self.value_head.forward(features)
        advantage = self.advantage.forward(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class AtariCNN(QNetwork):
    """The CNN used in the 2015 Mhin et al. DQN paper"""

    def __init__(self, input_shape: tuple[int, ...], extras_shape: tuple[int, ...], output_shape: tuple[int, ...]):
        assert len(input_shape) == 3
        assert len(output_shape) == 1
        super().__init__(input_shape, extras_shape, output_shape)
        filters = [32, 64, 64]
        kernels = [8, 4, 3]
        strides = [4, 2, 1]
        self.cnn, n_features = make_cnn(input_shape, filters, kernels, strides)
        self.linear = torch.nn.Sequential(torch.nn.Linear(n_features, 512), torch.nn.ReLU(), torch.nn.Linear(512, output_shape[0]))

    def forward(self, obs: torch.Tensor, extras: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, n_agents, channels, height, width = obs.shape
        obs = obs.view(batch_size * n_agents, channels, height, width)
        qvalues: torch.Tensor = self.cnn.forward(obs)
        return qvalues.view(batch_size, n_agents, -1)


class CNN(QNetwork):
    """
    CNN with three convolutional layers. The CNN output (output_cnn) is flattened and the extras are
    concatenated to this output. The CNN is followed by three linear layers (512, 256, output_shape[0]).
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        extras_size: int,
        output_shape: tuple[int, ...],
        mlp_sizes: tuple[int, ...] = (64, 64),
    ):
        assert len(input_shape) == 3, f"CNN can only handle 3D input shapes ({len(input_shape)} here)"
        super().__init__(input_shape, (extras_size,), output_shape)

        kernel_sizes = [3, 3, 3]
        strides = [1, 1, 1]
        filters = [32, 64, 64]
        self.cnn, n_features = make_cnn(self.input_shape, filters, kernel_sizes, strides)
        self.linear = MLP(n_features, extras_size, mlp_sizes, output_shape)

    @classmethod
    def from_env(cls, env: RLEnv, mlp_sizes: tuple[int, ...] = (64, 64)):
        return cls(env.observation_shape, env.extra_feature_shape[0], (env.n_actions, env.reward_size), mlp_sizes)

    def forward(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        # For transitions, the shape is (batch_size, n_agents, channels, height, width)
        # For episodes, the shape is (time, batch_size, n_agents, channels, height, width)
        *dims, channels, height, width = obs.shape
        bs = math.prod(dims)
        obs = obs.view(bs, channels, height, width)
        features = self.cnn.forward(obs)
        extras = extras.view(bs, *self.extras_shape)
        res = self.linear.forward(features, extras)
        return res.view(*dims, *self.output_shape)


class CNN_ActorCritic(ActorCriticNN):
    def __init__(self, input_shape: tuple[int, int, int], extras_shape: tuple[int], output_shape: tuple[int]):
        assert len(input_shape) == 3, f"CNN can only handle 3D input shapes ({len(input_shape)} here)"
        assert extras_shape is None or len(extras_shape) == 1, f"CNN can only handle 1D extras shapes ({len(extras_shape)} here)"
        assert len(output_shape) == 1, f"CNN can only handle 1D input shapes ({len(output_shape)} here)"
        super().__init__(input_shape, extras_shape, output_shape)

        kernel_sizes = [3, 3, 3]
        strides = [1, 1, 1]
        filters = [32, 64, 64]

        self.cnn, n_features = make_cnn(self.input_shape, filters, kernel_sizes, strides)
        common_input_size = n_features + self.extras_shape[0]
        self.common = torch.nn.Sequential(
            torch.nn.Linear(common_input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
        )

        self.policy_network = torch.nn.Sequential(
            torch.nn.Linear(128, *output_shape),
            # torch.nn.Softmax(dim=-1), use logits to mask invalid actions
        )
        self.value_network = torch.nn.Linear(128, 1)

    def _cnn_forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Check that the input has the correct shape (at least 4 dimensions)
        *dims, channels, height, width = obs.shape
        obs = obs.view(-1, channels, height, width)
        features = self.cnn.forward(obs)
        return features

    def policy(self, obs: torch.Tensor):
        return self.policy_network(obs)

    def value(self, obs: torch.Tensor):
        return self.value_network(obs)

    def forward(self, obs: torch.Tensor, extras: torch.Tensor):
        features = self._cnn_forward(obs)
        extras = extras.view(-1, *self.extras_shape)
        features = torch.cat((features, extras), dim=-1)
        x = self.common(features)
        return self.policy(x), self.value(x)

    @property
    def value_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.cnn.parameters()) + list(self.common.parameters()) + list(self.value_network.parameters())

    @property
    def policy_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.cnn.parameters()) + list(self.common.parameters()) + list(self.policy_network.parameters())


class SimpleActorCritic(ActorCriticNN):
    def __init__(self, input_shape: tuple[int], extras_shape: tuple[int, ...], output_shape: tuple[int]):
        super().__init__(input_shape, extras_shape, output_shape)
        self.common = torch.nn.Sequential(
            torch.nn.Linear(*input_shape, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
        )
        self.policy_network = torch.nn.Sequential(
            torch.nn.Linear(256, *output_shape),
            torch.nn.Softmax(dim=-1),
        )

        self.value_network = torch.nn.Linear(256, 1)

    def forward(self, obs: torch.Tensor, extras: torch.Tensor):
        obs = torch.cat((obs, extras), dim=-1)
        x = self.common(obs)
        return self.policy_network(x), self.value_network(x)

    def policy(self, obs: torch.Tensor):
        return self.policy_network(obs)

    def value(self, obs: torch.Tensor):
        return self.value_network(obs)

    @property
    def value_parameters(self):
        return list(self.common.parameters()) + list(self.value_network.parameters())

    @property
    def policy_parameters(self):
        return list(self.common.parameters()) + list(self.policy_network.parameters())


class PolicyNetworkMLP(NN):
    def __init__(self, input_shape: tuple[int, ...], extras_shape: tuple[int, ...], output_shape: tuple[int, ...]):
        super().__init__(input_shape, extras_shape, output_shape)
        assert len(self.extras_shape) == 1 and len(output_shape) == 1 and len(input_shape) == 1
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(input_shape[0] + self.extras_shape[0], 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_shape[0]),
            torch.nn.Softmax(dim=-1),
        )

    def forward(self, obs: torch.Tensor, extras: torch.Tensor | None = None) -> torch.Tensor:
        if extras is not None:
            obs = torch.cat((obs, extras), dim=-1)
        return self.nn.forward(obs)


def make_cnn(input_shape, filters: list[int], kernel_sizes: list[int], strides: list[int]):
    """Create a CNN with flattened output based on the given filters, kernel sizes and strides."""
    channels, height, width = input_shape
    paddings = [0 for _ in filters]
    n_padded = 0
    output_w, output_h = conv2d_size_out(width, height, kernel_sizes, strides, paddings)
    while output_w < 0 or output_h < 0:
        # Add paddings if the output size is negative
        paddings[n_padded % len(paddings)] += 1
        n_padded += 1
        output_w, output_h = conv2d_size_out(width, height, kernel_sizes, strides, paddings)
    assert output_h > 0 and output_w > 0, f"Input size = {input_shape}, output witdh = {output_w}, output height = {output_h}"
    modules = []
    for f, k, s, p in zip(filters, kernel_sizes, strides, paddings):
        modules.append(torch.nn.Conv2d(in_channels=channels, out_channels=f, kernel_size=k, stride=s, padding=p))
        modules.append(torch.nn.ReLU())
        channels = f
    modules.append(torch.nn.Flatten())
    output_size = output_h * output_w * filters[-1]
    return torch.nn.Sequential(*modules), output_size


def conv2d_size_out(input_width: int, input_height: int, kernel_sizes: list[int], strides: list[int], paddings: list[int]):
    """
    Compute the output width and height of a sequence of 2D convolutions.
    See shape section on https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """
    width = input_width
    height = input_height
    for kernel_size, stride, pad in zip(kernel_sizes, strides, paddings):
        width = (width + 2 * pad - (kernel_size - 1) - 1) // stride + 1
        height = (height + 2 * pad - (kernel_size - 1) - 1) // stride + 1
    return width, height


class ACNetwork(ActorCriticNN):
    def __init__(self, input_shape: tuple[int], extras_shape: tuple, output_shape: tuple):
        super().__init__(input_shape, extras_shape, output_shape)
        print(input_shape)
        self.common = torch.nn.Sequential(
            torch.nn.Linear(*input_shape, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
        )
        self.policy_network = torch.nn.Linear(128, *output_shape)
        self.value_network = torch.nn.Linear(128, 1)

    def policy(self, obs: torch.Tensor):
        obs = self.common(obs)
        return self.policy_network(obs)

    def value(self, obs: torch.Tensor):
        obs = self.common(obs)
        return self.value_network(obs)

    def forward(self, x):
        x = self.common(x)
        return self.policy_network(x), self.value_network(x)

    @property
    def value_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.common.parameters()) + list(self.value_network.parameters())

    @property
    def policy_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.common.parameters()) + list(self.policy_network.parameters())


class ACNetwork2(ActorCriticNN):
    def __init__(self, input_shape: tuple[int], extras_shape: tuple, output_shape: tuple):
        super().__init__(input_shape, extras_shape, output_shape)
        self.value_network = torch.nn.Sequential(
            torch.nn.Linear(*input_shape, 128), torch.nn.ReLU(), torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1)
        )
        self.policy_network = torch.nn.Sequential(
            torch.nn.Linear(*input_shape, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, *output_shape),
        )

    def policy(self, obs: torch.Tensor):
        return self.policy_network(obs)

    def value(self, obs: torch.Tensor):
        return self.value_network(obs)

    def forward(self, x):
        return self.policy_network(x), self.value_network(x)

    @property
    def value_parameters(self):
        return self.value_network.parameters()

    @property
    def policy_parameters(self):
        return self.policy_network.parameters()
