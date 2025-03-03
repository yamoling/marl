import math
import operator
from dataclasses import dataclass
from functools import reduce
from typing import Optional, Sequence

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from marlenv import ActionSpace, Observation, DiscreteActionSpace
from marlenv.models import MARLEnv

from marl.agents.qlearning.maic import MAICParameters
from marl.models.nn import MAIC, QNetwork, RecurrentQNetwork, MAICNN
from ..utils import make_cnn


@dataclass(unsafe_hash=True)
class MLP(QNetwork):
    """
    Multi layer perceptron
    """

    layer_sizes: Sequence[int]

    def __init__(
        self,
        input_size: int,
        extras_size: int,
        hidden_sizes: Sequence[int],
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
    def from_env[A, AS: ActionSpace](cls, env: MARLEnv[A, AS], hidden_sizes: Optional[Sequence[int]] = None):
        if env.is_multi_objective:
            output_shape = (env.n_actions, env.reward_space.size)
        else:
            output_shape = (env.n_actions,)
        if hidden_sizes is None:
            hidden_sizes = (64,)
        return cls(
            env.observation_shape[0],
            env.extras_shape[0],
            tuple(hidden_sizes),
            output_shape,
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
    def from_env(cls, env: MARLEnv, nn: QNetwork):
        assert nn.input_shape == env.observation_shape
        assert nn.extras_shape == env.extras_shape
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


@dataclass(unsafe_hash=True)
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
    def from_env[A](cls, env: MARLEnv[A, DiscreteActionSpace], mlp_sizes: tuple[int, ...] = (64, 64)):
        if env.reward_space.size == 1:
            output_shape = (env.n_actions,)
        else:
            output_shape = (env.n_actions, env.reward_space.size)
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
        extras = extras.reshape(bs, *self.extras_shape)
        res = self.linear.forward(features, extras)
        return res.view(*dims, *self.output_shape)


class IndependentCNN(QNetwork):
    """
    CNN with three convolutional layers. The CNN output (output_cnn) is flattened and the extras are
    concatenated to this output. The CNN is followed by three linear layers (512, 256, output_shape[0]).
    """

    def __init__(
        self,
        n_agents: int,
        input_shape: tuple[int, ...],
        extras_size: int,
        output_shape: tuple[int, ...],
        mlp_sizes: tuple[int, ...] = (64, 64),
    ):
        assert len(input_shape) == 3, f"CNN can only handle 3D input shapes ({len(input_shape)} here)"
        super().__init__(input_shape, (extras_size,), output_shape)
        self.n_agents = n_agents
        kernel_sizes = [3, 3, 3]
        strides = [1, 1, 1]
        filters = [32, 64, 64]
        self.cnn, n_features = make_cnn(self.input_shape, filters, kernel_sizes, strides)
        self.layer_sizes = (n_features + extras_size, *mlp_sizes, math.prod(output_shape))
        linears = []
        for _ in range(n_agents):
            layers = [torch.nn.Linear(n_features + extras_size, mlp_sizes[0]), torch.nn.ReLU()]
            for i in range(len(mlp_sizes) - 1):
                layers.append(torch.nn.Linear(mlp_sizes[i], mlp_sizes[i + 1]))
                layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(mlp_sizes[-1], math.prod(output_shape)))
            linears.append(torch.nn.Sequential(*layers))
        self.linears = torch.nn.ModuleList(linears)

    def to(self, device: torch.device, dtype: torch.dtype | None = None, non_blocking=True):
        self.linears.to(device, dtype, non_blocking)
        return super().to(device, dtype, non_blocking)

    @classmethod
    def from_env(cls, env: MARLEnv, mlp_sizes: tuple[int, ...] = (64, 64)):
        if env.is_multi_objective:
            output_shape = (env.n_actions, env.reward_space.size)
        else:
            output_shape = (env.n_actions,)
        return cls(env.n_agents, env.observation_shape, env.extras_shape[0], output_shape, mlp_sizes)

    def forward(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        # For transitions, the shape is (batch_size, n_agents, channels, height, width)
        # For episodes, the shape is (time, batch_size, n_agents, channels, height, width)
        batch_size, n_agents, channels, height, width = obs.shape
        # Transpose to (batch_size, n_agents, channels, height, width)
        obs = obs.transpose(0, 1)
        extras = extras.transpose(0, 1)
        # Reshape to be able forward the CNN
        obs = obs.reshape(-1, channels, height, width)
        features = self.cnn.forward(obs)
        # Reshape to retrieve the 'agent' dimension
        features = torch.reshape(features, (n_agents, batch_size, -1))
        features = torch.concatenate((features, extras), dim=-1)
        res = []
        for agent_feature, linear in zip(features, self.linears):
            res.append(linear.forward(agent_feature))
        res = torch.stack(res)
        res = res.transpose(0, 1)
        return res.reshape(batch_size, n_agents, *self.output_shape)


class RCNN(RecurrentQNetwork):
    """
    Recurrent CNN.
    """

    def __init__(self, input_shape: tuple[int, int, int], extras_size: int, output_shape: tuple[int, ...]):
        super().__init__(input_shape, (extras_size,), output_shape)

        kernel_sizes = [3, 3, 3]
        strides = [1, 1, 1]
        filters = [32, 64, 64]
        self.cnn, self.n_features = make_cnn(self.input_shape, filters, kernel_sizes, strides)
        self.rnn = RNNQMix((self.n_features,), (extras_size,), output_shape)

    @classmethod
    def from_env(cls, env: MARLEnv):
        if env.is_multi_objective:
            output_shape = (env.n_actions, env.reward_space.size)
        else:
            output_shape = (env.n_actions,)
        assert len(env.observation_shape) == 3
        return cls(env.observation_shape, env.extras_shape[0], output_shape)

    def forward(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        # For transitions, the shape is (batch_size, n_agents, channels, height, width)
        # For episodes, the shape is (time, batch_size, n_agents, channels, height, width)
        *dims, channels, height, width = obs.shape
        obs = obs.view(-1, channels, height, width)
        features = self.cnn.forward(obs)
        features = torch.reshape(features, (*dims, self.n_features))
        extras = extras.view(*dims, *self.extras_shape)
        res = self.rnn.forward(features, extras)
        return res.view(*dims, *self.output_shape)

    def batch_forward(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        *dims, channels, height, width = obs.shape
        obs = obs.reshape(-1, channels, height, width)
        features = self.cnn.forward(obs)
        features = torch.reshape(features, (*dims, self.n_features))
        extras = extras.view(*dims, *self.extras_shape)
        res = self.rnn.batch_forward(features, extras)
        return res.view(*dims, *self.output_shape)

    def value(self, obs: Observation) -> torch.Tensor:
        x, extras = self.to_tensor(obs)
        *dims, channels, height, width = x.shape
        x = x.view(-1, channels, height, width)
        features = self.cnn.forward(x).unsqueeze(0)
        extras = extras.view(*dims, *self.extras_shape)
        saved_hidden_states = self.rnn.hidden_states
        qvalues = self.rnn.forward(features, extras)
        self.rnn.hidden_states = saved_hidden_states
        max_qvalues = qvalues.max(dim=-2).values
        return max_qvalues.mean(dim=-2)


class MAICNetworkRDQN(RecurrentQNetwork, MAIC):
    """
    Source : https://github.com/mansicer/MAIC
    """

    def __init__(self, input_shape: tuple[int, ...], extras_shape: tuple[int, ...], output_size: int, args: MAICParameters):
        super().__init__(input_shape, extras_shape, (output_size,))

        self.args = args
        self.n_agents = args.n_agents
        self.latent_dim = args.latent_dim
        self.n_actions = output_size

        self.test_mode = False

        if self.args.com:
            NN_HIDDEN_SIZE = args.nn_hidden_size
            activation_func = nn.LeakyReLU()

            self.embed_net = nn.Sequential(
                nn.Linear(args.rnn_hidden_dim, NN_HIDDEN_SIZE),
                nn.BatchNorm1d(NN_HIDDEN_SIZE),
                activation_func,
                nn.Linear(NN_HIDDEN_SIZE, args.n_agents * args.latent_dim * 2),
            )

            self.inference_net = nn.Sequential(
                nn.Linear(args.rnn_hidden_dim + self.n_actions, NN_HIDDEN_SIZE),
                nn.BatchNorm1d(NN_HIDDEN_SIZE),
                activation_func,
                nn.Linear(NN_HIDDEN_SIZE, args.latent_dim * 2),
            )
            self.msg_net = nn.Sequential(
                nn.Linear(args.rnn_hidden_dim + args.latent_dim, NN_HIDDEN_SIZE), activation_func, nn.Linear(NN_HIDDEN_SIZE, self.n_actions)
            )

            self.w_query = nn.Linear(args.rnn_hidden_dim, args.attention_dim)
            self.w_key = nn.Linear(args.latent_dim, args.attention_dim)

        n_inputs = reduce(operator.mul, input_shape) + extras_shape[0]

        self.fc1 = nn.Linear(n_inputs, args.rnn_hidden_dim)
        self.rnn = nn.GRU(args.rnn_hidden_dim, args.rnn_hidden_dim, batch_first=False)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, self.n_actions)

    def _compute_messages(self, x, bs):
        latent_parameters = self.embed_net(x)
        latent_parameters[:, -self.n_agents * self.latent_dim :] = torch.clamp(
            torch.exp(latent_parameters[:, -self.n_agents * self.latent_dim :]), min=self.args.var_floor
        )

        latent_embed = latent_parameters.reshape(bs * self.n_agents, self.n_agents * self.latent_dim * 2)

        if self.test_mode:
            latent = latent_embed[:, : self.n_agents * self.latent_dim]
        else:
            gaussian_embed = D.Normal(
                latent_embed[:, : self.n_agents * self.latent_dim], (latent_embed[:, self.n_agents * self.latent_dim :]) ** (1 / 2)
            )
            latent = gaussian_embed.rsample()  # shape: (bs * self.n_agents, self.n_agents * self.latent_dim)
        latent = latent.reshape(bs * self.n_agents * self.n_agents, self.latent_dim)

        h_repeat = x.view(bs, self.n_agents, -1).repeat(1, self.n_agents, 1).view(bs * self.n_agents * self.n_agents, -1)
        msg = self.msg_net(torch.cat([h_repeat, latent], dim=-1)).view(bs, self.n_agents, self.n_agents, self.n_actions)

        query = self.w_query(x).unsqueeze(1)
        key = self.w_key(latent).reshape(bs * self.n_agents, self.n_agents, -1).transpose(1, 2)
        alpha = torch.bmm(query / (self.args.attention_dim ** (1 / 2)), key).view(bs, self.n_agents, self.n_agents)
        for i in range(self.n_agents):
            alpha[:, i, i] = -1e9
        alpha = F.softmax(alpha, dim=-1).reshape(bs, self.n_agents, self.n_agents, 1)

        if self.test_mode:
            alpha[alpha < (0.25 * 1 / self.n_agents)] = 0

        gated_msg = alpha * msg

        return gated_msg

    def get_values_and_comms(self, obs: torch.Tensor, extras: torch.Tensor):
        *dims, channels, height, width = obs.shape

        is_batch = len(dims) == 3  # episode batch ?
        total_batch = math.prod(dims)

        bs = math.prod(dims[:-1]) if is_batch else 1

        obs = obs.reshape(total_batch, -1)
        if extras is not None:
            extras = extras.reshape(total_batch, *self.extras_shape)
            obs = torch.concat((obs, extras), dim=-1)

        x = F.relu(self.fc1(obs))
        x, self.hidden_states = self.rnn(x, self.hidden_states)
        q = self.fc2(x)

        messages = []
        gated_msg = None
        init_qvalues = q.detach().clone()

        if self.args.com:
            gated_msg = self._compute_messages(x, bs)
            messages = torch.sum(gated_msg, dim=1).view(bs * self.n_agents, self.n_actions)
            q += messages

        return q.view(*dims, *self.output_shape).unsqueeze(-1), gated_msg, messages, init_qvalues

    def forward(self, obs: torch.Tensor, extras: torch.Tensor):
        q_values, _, _, _ = self.get_values_and_comms(obs, extras)
        return q_values

    @classmethod
    def from_env[A](cls, env: MARLEnv[A, DiscreteActionSpace], args: MAICParameters):
        return cls(env.observation_shape, env.extras_shape, env.n_actions, args)


class MAICNetworkCNN(MAICNN):
    """
    Source : https://github.com/mansicer/MAIC
    """

    def __init__(self, input_shape: tuple[int, ...], extras_shape: tuple[int, ...], output_size: int, args: MAICParameters):
        assert len(input_shape) == 3, f"CNN can only handle 3D input shapes ({len(input_shape)} here)"
        super().__init__(input_shape, extras_shape, (output_size,))

        kernel_sizes = [3, 3, 3]
        strides = [1, 1, 1]
        filters = [32, 64, 64]

        self.args = args
        self.n_agents = args.n_agents
        self.latent_dim = args.latent_dim
        self.n_actions = output_size

        self.test_mode = False

        if self.args.com:
            NN_HIDDEN_SIZE = args.nn_hidden_size
            activation_func = nn.LeakyReLU()

            self.embed_net = nn.Sequential(
                nn.Linear(args.rnn_hidden_dim, NN_HIDDEN_SIZE),
                nn.BatchNorm1d(NN_HIDDEN_SIZE),
                activation_func,
                nn.Linear(NN_HIDDEN_SIZE, args.n_agents * args.latent_dim * 2),
            )

            self.inference_net = nn.Sequential(
                nn.Linear(args.rnn_hidden_dim + self.n_actions, NN_HIDDEN_SIZE),
                nn.BatchNorm1d(NN_HIDDEN_SIZE),
                activation_func,
                nn.Linear(NN_HIDDEN_SIZE, args.latent_dim * 2),
            )
            self.msg_net = nn.Sequential(
                nn.Linear(args.rnn_hidden_dim + args.latent_dim, NN_HIDDEN_SIZE), activation_func, nn.Linear(NN_HIDDEN_SIZE, self.n_actions)
            )

            self.w_query = nn.Linear(args.rnn_hidden_dim, args.attention_dim)
            self.w_key = nn.Linear(args.latent_dim, args.attention_dim)

        self.cnn, n_features = make_cnn(self.input_shape, filters, kernel_sizes, strides)
        cnn_output_dim = n_features + self.extras_shape[0]

        self.fc1 = nn.Linear(cnn_output_dim, args.rnn_hidden_dim)  # TODO: rename the parameter
        self.fc2 = nn.Linear(args.rnn_hidden_dim, self.n_actions)

    def _compute_messages(self, x, bs):
        latent_parameters = self.embed_net(x)
        latent_parameters[:, -self.n_agents * self.latent_dim :] = torch.clamp(
            torch.exp(latent_parameters[:, -self.n_agents * self.latent_dim :]), min=self.args.var_floor
        )

        latent_embed = latent_parameters.reshape(bs * self.n_agents, self.n_agents * self.latent_dim * 2)

        if self.test_mode:
            latent = latent_embed[:, : self.n_agents * self.latent_dim]
        else:
            gaussian_embed = D.Normal(
                latent_embed[:, : self.n_agents * self.latent_dim], (latent_embed[:, self.n_agents * self.latent_dim :]) ** (1 / 2)
            )
            latent = gaussian_embed.rsample()  # shape: (bs * self.n_agents, self.n_agents * self.latent_dim)
        latent = latent.reshape(bs * self.n_agents * self.n_agents, self.latent_dim)

        h_repeat = x.view(bs, self.n_agents, -1).repeat(1, self.n_agents, 1).view(bs * self.n_agents * self.n_agents, -1)
        msg = self.msg_net(torch.cat([h_repeat, latent], dim=-1)).view(bs, self.n_agents, self.n_agents, self.n_actions)

        query = self.w_query(x).unsqueeze(1)
        key = self.w_key(latent).reshape(bs * self.n_agents, self.n_agents, -1).transpose(1, 2)
        alpha = torch.bmm(query / (self.args.attention_dim ** (1 / 2)), key).view(bs, self.n_agents, self.n_agents)
        for i in range(self.n_agents):
            alpha[:, i, i] = -1e9
        alpha = F.softmax(alpha, dim=-1).reshape(bs, self.n_agents, self.n_agents, 1)

        if self.test_mode:
            alpha[alpha < (0.25 * 1 / self.n_agents)] = 0

        gated_msg = alpha * msg

        return torch.sum(gated_msg, dim=1).view(bs * self.n_agents, self.n_actions)

    def forward(self, obs: torch.Tensor, extras: torch.Tensor):
        *dims, channels, height, width = obs.shape

        is_batch = len(dims) == 3  # episode batch ?
        total_batch = math.prod(dims)

        bs = math.prod(dims[:-1]) if is_batch else 1

        obs = obs.reshape(total_batch, channels, height, width)
        if extras is not None:
            extras = extras.reshape(total_batch, *self.extras_shape)

        x = F.relu(self.cnn(obs))

        if extras is not None:
            x = torch.concat((x, extras), dim=-1)

        x = F.relu(self.fc1(x))
        q = self.fc2(x)

        if self.args.com:
            q += self._compute_messages(x, bs)

        return q.view(*dims, *self.output_shape).unsqueeze(-1)

    @classmethod
    def from_env[A](cls, env: MARLEnv[A, DiscreteActionSpace], args: MAICParameters):
        return cls(env.observation_shape, env.extras_shape, env.n_actions, args)


class MAICNetworkCNNRDQN(RecurrentQNetwork, MAIC):
    """
    Source : https://github.com/mansicer/MAIC
    """

    def __init__(self, input_shape: tuple[int, ...], extras_shape: tuple[int, ...], output_size: int, args: MAICParameters):
        assert len(input_shape) == 3, f"CNN can only handle 3D input shapes ({len(input_shape)} here)"
        super().__init__(input_shape, extras_shape, (output_size,))

        kernel_sizes = [3, 3, 3]
        strides = [1, 1, 1]
        filters = [32, 64, 64]

        self.args = args
        self.n_agents = args.n_agents
        self.latent_dim = args.latent_dim
        self.n_actions = output_size

        self.test_mode = False

        if self.args.com:
            NN_HIDDEN_SIZE = args.nn_hidden_size
            activation_func = nn.LeakyReLU()

            self.embed_net = nn.Sequential(
                nn.Linear(args.rnn_hidden_dim, NN_HIDDEN_SIZE),
                nn.BatchNorm1d(NN_HIDDEN_SIZE),
                activation_func,
                nn.Linear(NN_HIDDEN_SIZE, args.n_agents * args.latent_dim * 2),
            )

            self.inference_net = nn.Sequential(
                nn.Linear(args.rnn_hidden_dim + self.n_actions, NN_HIDDEN_SIZE),
                nn.BatchNorm1d(NN_HIDDEN_SIZE),
                activation_func,
                nn.Linear(NN_HIDDEN_SIZE, args.latent_dim * 2),
            )
            self.msg_net = nn.Sequential(
                nn.Linear(args.rnn_hidden_dim + args.latent_dim, NN_HIDDEN_SIZE), activation_func, nn.Linear(NN_HIDDEN_SIZE, self.n_actions)
            )

            self.w_query = nn.Linear(args.rnn_hidden_dim, args.attention_dim)
            self.w_key = nn.Linear(args.latent_dim, args.attention_dim)

        self.cnn, n_features = make_cnn(self.input_shape, filters, kernel_sizes, strides)
        cnn_output_dim = n_features + self.extras_shape[0]

        self.fc1 = nn.Linear(cnn_output_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRU(args.rnn_hidden_dim, args.rnn_hidden_dim, batch_first=False)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, self.n_actions)

    def _compute_messages(self, x, bs):
        latent_parameters = self.embed_net(x)
        latent_parameters[:, -self.n_agents * self.latent_dim :] = torch.clamp(
            torch.exp(latent_parameters[:, -self.n_agents * self.latent_dim :]), min=self.args.var_floor
        )

        latent_embed = latent_parameters.reshape(bs * self.n_agents, self.n_agents * self.latent_dim * 2)

        if self.test_mode:
            latent = latent_embed[:, : self.n_agents * self.latent_dim]
        else:
            gaussian_embed = D.Normal(
                latent_embed[:, : self.n_agents * self.latent_dim], (latent_embed[:, self.n_agents * self.latent_dim :]) ** (1 / 2)
            )
            latent = gaussian_embed.rsample()  # shape: (bs * self.n_agents, self.n_agents * self.latent_dim)
        latent = latent.reshape(bs * self.n_agents * self.n_agents, self.latent_dim)

        h_repeat = x.view(bs, self.n_agents, -1).repeat(1, self.n_agents, 1).view(bs * self.n_agents * self.n_agents, -1)
        msg = self.msg_net(torch.cat([h_repeat, latent], dim=-1)).view(bs, self.n_agents, self.n_agents, self.n_actions)

        query = self.w_query(x).unsqueeze(1)
        key = self.w_key(latent).reshape(bs * self.n_agents, self.n_agents, -1).transpose(1, 2)
        alpha = torch.bmm(query / (self.args.attention_dim ** (1 / 2)), key).view(bs, self.n_agents, self.n_agents)
        for i in range(self.n_agents):
            alpha[:, i, i] = -1e9
        alpha = F.softmax(alpha, dim=-1).reshape(bs, self.n_agents, self.n_agents, 1)

        if self.test_mode:
            alpha[alpha < (0.25 * 1 / self.n_agents)] = 0

        gated_msg = alpha * msg

        return gated_msg

    def get_values_and_comms(self, obs: torch.Tensor, extras: torch.Tensor):
        *dims, channels, height, width = obs.shape

        is_batch = len(dims) == 3  # episode batch ?
        total_batch = math.prod(dims)

        bs = math.prod(dims[:-1]) if is_batch else 1

        obs = obs.reshape(total_batch, channels, height, width)
        if extras is not None:
            extras = extras.reshape(total_batch, *self.extras_shape)

        x = F.relu(self.cnn(obs))

        if extras is not None:
            x = torch.concat((x, extras), dim=-1)

        x = F.relu(self.fc1(x))
        x, self.hidden_states = self.rnn(x, self.hidden_states)
        q = self.fc2(x)

        messages = []
        gated_msg = None
        init_qvalues = q.detach().clone()

        if self.args.com:
            gated_msg = self._compute_messages(x, bs)
            messages = torch.sum(gated_msg, dim=1).view(bs * self.n_agents, self.n_actions)
            q += messages

        return q.view(*dims, *self.output_shape).unsqueeze(-1), gated_msg, messages, init_qvalues

    def forward(self, obs: torch.Tensor, extras: torch.Tensor):
        q_values, _, _, _ = self.get_values_and_comms(obs, extras)
        return q_values

    @classmethod
    def from_env[A](cls, env: MARLEnv[A, DiscreteActionSpace], args: MAICParameters):
        return cls(env.observation_shape, env.extras_shape, env.n_actions, args)
