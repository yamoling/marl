import math
import operator
from abc import abstractmethod
from dataclasses import KW_ONLY, dataclass, field
from functools import reduce
from typing import TYPE_CHECKING, Literal, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from marlenv import MARLEnv, MultiDiscreteSpace
from torch import Tensor, distributions
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from marl.models.nn import MAIC, MAICNN, NN, QNetwork, RecurrentQNetwork

from ..layers import NoisyLinear
from ..utils import make_cnn
from .generic import CNN, MLP

if TYPE_CHECKING:
    from marl.training.maic import MAICParameters


@dataclass(unsafe_hash=True)
class QCNN(CNN, QNetwork):
    def __post_init__(self):
        QNetwork.__post_init__(self)
        CNN.__post_init__(self)

    @classmethod
    def from_env(cls, env: MARLEnv[MultiDiscreteSpace], mlp_sizes: Sequence[int] = (128, 128), mlp_noisy: bool = False):
        if env.is_multi_objective:
            output_shape = (env.n_actions, env.reward_space.size)
        else:
            output_shape = (env.n_actions,)
        assert len(env.observation_shape) == 3
        return QCNN(output_shape, env.observation_shape, env.extras_size, mlp_sizes, mlp_noisy)


@dataclass(unsafe_hash=True)
class QMLP(MLP, QNetwork):
    def __post_init__(self):
        QNetwork.__post_init__(self)
        MLP.__post_init__(self)

    @classmethod
    def from_env(cls, env: MARLEnv[MultiDiscreteSpace], mlp_sizes: Sequence[int] = (128, 128), mlp_noisy: bool = False):
        if env.is_multi_objective:
            output_shape = (env.n_actions, env.reward_space.size)
        else:
            output_shape = (env.n_actions,)
        assert len(env.observation_shape) == 1
        return QMLP(output_shape, env.observation_shape[0], env.extras_size, mlp_sizes, noisy=mlp_noisy)

    # def __init__(
    #     self,
    #     input_size: int,
    #     extras_size: int,
    #     output_shape: int | tuple[int, int],
    #     hidden_sizes: Sequence[int] = (128, 128),
    #     last_layer_noisy: bool = False,
    # ):
    #     if isinstance(output_shape, int):
    #         output = (output_shape,)
    #     else:
    #         output = output_shape
    #     QNetwork.__init__(self, output)
    #     MLP.__init__(self, input_size, extras_size, output, hidden_sizes, noisy=last_layer_noisy)


@dataclass(unsafe_hash=True)
class RNNQMix(RecurrentQNetwork):
    """RNN used in the QMix paper:
    - linear 64
    - relu
    - GRU 64
    - relu
    - linear 64"""

    obs_size: int
    extras_size: int

    @property
    def input_size(self):
        return self.obs_size + self.extras_size

    def __post_init__(self):
        super().__post_init__()
        self.fc1 = torch.nn.Sequential(torch.nn.Linear(self.input_size, 64), torch.nn.ReLU())
        self.gru = torch.nn.GRU(input_size=64, hidden_size=64, batch_first=False)
        self.fc2 = torch.nn.Linear(64, self.output_size)

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, /, masks: torch.Tensor | None = None, **kwargs):
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
            x = self.fc1.forward(x)
            if masks is not None:
                episode_lengths = masks.sum(0).long()
                x = pack_padded_sequence(x, episode_lengths.cpu(), enforce_sorted=False)
                x, self._hidden_states = self.gru.forward(x, self._hidden_states)
                x, _ = pad_packed_sequence(x)
            else:
                x, self._hidden_states = self.gru.forward(x, self._hidden_states)
            x = self.fc2.forward(x)
            # Restore the original shape of the batch
            x = x.view(episode_length, *batch_agents, *self.output_shape)
            return x
        except RuntimeError as e:
            error_message = str(e)
            if "shape" in error_message:
                error_message += "\nDid you use a TransitionMemory instead of an EpisodeMemory alongside an RNN ?"
            raise RuntimeError(error_message)


@dataclass(unsafe_hash=True)
class MAVENTail(torch.nn.Module):
    """
    Tail of the MAVEN agent-wise network. The paper only presents the "hyper-network" approach
    but the official code has two kinds of networks.
    """

    noise_size: int
    n_agents: int
    agent_output_size: int
    n_actions: int

    def __post_init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, noise: Tensor, agent_output: Tensor) -> Tensor: ...


@dataclass(unsafe_hash=True)
class MAVENHyperBMM(MAVENTail):
    """
    This tail network is the approach presented in the MAVEN paper, i.e. a hyper-network that generates the weights to compute the q-values directly from the noise and agent ids.
    """

    def __post_init__(self):
        super().__post_init__()
        self.hyper_network = torch.nn.Linear(
            self.noise_size + self.n_agents,
            self.agent_output_size * self.n_actions,
        )

    def forward(self, noise: Tensor, agent_output: Tensor) -> Tensor:
        """
        The hyper-network takes as input the noise and the agent id and produces the weight matrix that will be multiplied with the previous layer outputs.
        The final output is of shape (batch_size, n_agents, n_actions), i.e. a q-value per agent and per action.
        """
        *dims, n_agents, noise_size = noise.shape
        batch_size = math.prod(dims)
        # Build the hyper-network inputs: [noise, agent_id]
        agent_ids = torch.eye(self.n_agents, device=noise.device).unsqueeze(0).repeat(batch_size, 1, 1)
        noise = noise.reshape(batch_size, n_agents, noise_size)
        inputs = torch.cat([noise, agent_ids], dim=-1)
        # The hyper-network takes as input the [noise, agent_id] and outputs HIDDEN_DIM * n_actions weights.
        weights = self.hyper_network.forward(inputs)
        # Reshape to match the batch matrix multiplication requirements
        # Agent_output: (batch_size, n_agents, agent_output) -> (batch_size * n_agents, 1, agent_output)
        # Weights     : (batch_size, n_agents, agent_output * n_actions) -> (batch_size * n_agents, agent_output, n_actions)
        weights = weights.view(batch_size * self.n_agents, self.agent_output_size, self.n_actions)
        agent_output = agent_output.view(batch_size * self.n_agents, 1, self.agent_output_size)
        res = torch.bmm(agent_output, weights)
        # Return in the original shape
        return res.view(*dims, self.n_agents, self.n_actions)


@dataclass(unsafe_hash=True)
class MAVENHyperMult(MAVENTail):
    def __post_init__(self):
        super().__post_init__()
        self.linear = torch.nn.Linear(self.agent_output_size, self.n_actions)
        self.mult_weights_nn = torch.nn.Sequential(
            torch.nn.Linear(self.noise_size + self.n_agents, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, self.n_actions),
        )

    def forward(self, noise: Tensor, agent_output: Tensor) -> Tensor:
        qs = self.linear.forward(agent_output)
        agent_ids = torch.eye(self.n_agents, device=noise.device)
        weights_inputs = torch.cat([noise, agent_ids])
        weights = self.mult_weights_nn.forward(weights_inputs)
        return qs * weights


@dataclass(unsafe_hash=True)
class MAVENNN(QNetwork):
    n_actions: int
    noise_size: int
    n_agents: int
    head: NN = field(init=False)
    _: KW_ONLY
    tail_type: Literal["bmm", "mul"] = "bmm"
    agent_output_size: int = 128

    def __post_init__(self):
        super().__post_init__()
        match self.tail_type:
            case "bmm":
                self.tail = MAVENHyperBMM(self.noise_size, self.n_agents, self.agent_output_size, self.n_actions)
            case "mul":
                self.tail = MAVENHyperMult(self.noise_size, self.n_agents, self.agent_output_size, self.n_actions)
            case other:
                raise ValueError(f"Unknown hyper network type {other}")

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, /, **kwargs) -> torch.Tensor:
        match len(extras.shape):
            case 3:
                noise = extras[:, :, -self.noise_size :]
                extras = extras[:, :, : -self.noise_size]
            case 4:
                noise = extras[:, :, :, -self.noise_size :]
                extras = extras[:, :, :, : -self.noise_size]
            case _:
                raise NotImplementedError()
        x = self.head.forward(obs, extras)
        return self.tail.forward(noise, x)


@dataclass(unsafe_hash=True)
class MAVENCNN(MAVENNN):
    obs_shape: tuple[int, int, int]
    extras_size: int

    def __post_init__(self):
        super().__post_init__()
        self.head = CNN(
            (self.agent_output_size,),
            self.obs_shape,
            self.extras_size - self.noise_size,
            mlp_sizes=(256, 256),
            output_activation="relu",
        )

    @classmethod
    def from_env(cls, env: MARLEnv[MultiDiscreteSpace], tail_type: Literal["bmm", "mul"] = "bmm"):
        assert len(env.observation_shape) == 3
        noise_size = len([m for m in env.extras_meanings if "noise" in m.lower() or "maven" in m.lower()])
        if noise_size == 0:
            raise ValueError(
                "No noise found in the environment extras. Make sure to add noise to the environment extras with env.extra_noise() or to use the MAVEN agent."
            )
        return MAVENCNN(
            (env.n_actions,),
            env.n_actions,
            noise_size,
            env.n_agents,
            env.observation_shape,
            env.extras_size,
            tail_type=tail_type,
        )


@dataclass(unsafe_hash=True)
class MAVENMLP(MAVENNN):
    extras_size: int
    obs_size: int

    def __post_init__(self):
        super().__post_init__()
        self.head = MLP((self.agent_output_size,), self.obs_size, self.extras_size - self.noise_size, (256, 256), noisy=False)

    @classmethod
    def from_env(cls, env: MARLEnv[MultiDiscreteSpace], tail_type: Literal["bmm", "mul"] = "bmm"):
        assert len(env.observation_shape) == 1
        noise_size = len([m for m in env.extras_meanings if "noise" in m.lower() or "maven" in m.lower()])
        if noise_size == 0:
            raise ValueError(
                "No noise found in the environment extras. Make sure to add noise to the environment extras with env.extra_noise() or to use the MAVEN agent."
            )
        return MAVENMLP(
            (env.n_actions,),
            env.n_actions,
            noise_size,
            env.n_agents,
            env.extras_size,
            env.observation_shape[0],
            tail_type=tail_type,
        )


RNN = RNNQMix


class AtariCNN(QNetwork):
    """The CNN used in the 2015 Mhin et al. DQN paper"""

    def __init__(self, input_shape: tuple[int, int, int], output_shape: int):
        super().__init__((output_shape,))
        filters = [32, 64, 64]
        kernels = [8, 4, 3]
        strides = [4, 2, 1]
        self.cnn, n_features = make_cnn(input_shape, filters, kernels, strides)
        self.linear = torch.nn.Sequential(torch.nn.Linear(n_features, 512), torch.nn.ReLU(), torch.nn.Linear(512, output_shape))

    def forward(self, obs: torch.Tensor, extras: Optional[torch.Tensor] = None, /, **kwargs) -> torch.Tensor:
        batch_size, n_agents, channels, height, width = obs.shape
        obs = obs.view(batch_size * n_agents, channels, height, width)
        qvalues: torch.Tensor = self.cnn.forward(obs)
        return qvalues.view(batch_size, n_agents, -1)


@dataclass(unsafe_hash=True)
class IndependentCNN(QNetwork):
    """
    CNN whose flattened output is concatenated with the extras to be fed to the linear layers.

    The CNN part of the network is shared but the linear layers are separated.
    """

    duelling: bool

    def __init__(
        self,
        n_agents: int,
        input_shape: tuple[int, int, int],
        extras_size: int,
        output_shape: tuple[int] | tuple[int, int],
        mlp_sizes: tuple[int, ...] = (64, 64),
        kernel_sizes: tuple[int, ...] = (3, 3, 3),
        strides: tuple[int, ...] = (1, 1, 1),
        filters: tuple[int, ...] = (32, 64, 64),
        duelling: bool = True,
        mlp_noisy: bool = False,
    ):
        super().__init__(output_shape)
        self.n_agents = n_agents
        assert len(strides) == len(filters) == len(kernel_sizes)
        self.cnn, n_features = make_cnn(input_shape, filters, kernel_sizes, strides)
        self.duelling = duelling
        n_outputs = math.prod(self.output_shape)
        if duelling:
            n_outputs += 1
        linears = []
        for _ in range(n_agents):
            layers: list[torch.nn.Module] = [torch.nn.Linear(n_features + extras_size, mlp_sizes[0]), torch.nn.ReLU()]
            for i in range(len(mlp_sizes) - 1):
                layers.append(torch.nn.Linear(mlp_sizes[i], mlp_sizes[i + 1]))
                layers.append(torch.nn.ReLU())
            if mlp_noisy:
                layers.append(NoisyLinear(mlp_sizes[-1], n_outputs))
            else:
                layers.append(torch.nn.Linear(mlp_sizes[-1], n_outputs))
            linears.append(torch.nn.Sequential(*layers))
        self.linears = torch.nn.ModuleList(linears)

    @classmethod
    def from_env(
        cls,
        env: MARLEnv[MultiDiscreteSpace],
        mlp_sizes: tuple[int, ...] = (64, 64),
        duelling: bool = True,
        mlp_noisy: bool = False,
    ):
        if env.is_multi_objective:
            output_shape = (env.n_actions, env.reward_space.size)
        else:
            output_shape = (env.n_actions,)
        assert len(env.observation_shape) == 3
        c, h, w = env.observation_shape
        return IndependentCNN(
            env.n_agents,
            (c, h, w),
            env.extras_shape[0],
            output_shape,
            mlp_sizes,
            duelling=duelling,
            mlp_noisy=mlp_noisy,
        )

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, /, **kwargs) -> torch.Tensor:
        # For transitions, the shape is (batch_size, n_agents, channels, height, width)
        # For episodes, the shape is (time, batch_size, n_agents, channels, height, width) -> Not implemented
        batch_size, n_agents, channels, height, width = obs.shape
        # Reshape to be able forward the CNN
        obs = obs.reshape(-1, channels, height, width)
        features = self.cnn.forward(obs)
        # Restore the batch dimension
        features = torch.reshape(features, (batch_size, n_agents, -1))
        features = torch.concatenate((features, extras), dim=-1)
        # Features have shape (batch_size, n_agents, ...) but we want to transpose to (n_agents, batch_size, ...)
        # such that each individual agent can process its batch.
        # Reshape to retrieve the 'agent' dimension
        features = features.transpose(0, 1)
        res = []
        for agent_feature, linear in zip(features, self.linears):
            res.append(linear.forward(agent_feature))
        res = torch.stack(res)
        if self.duelling:
            value = torch.unsqueeze(res[:, :, -1], -1)  # Unsqueeze to keep 3 dimensions (batch_size, n_agents, 1)
            adv = res[:, :, :-1]
            mean_adv = torch.mean(adv, dim=-1, keepdim=True)
            res = value + adv - mean_adv
        res = res.transpose(0, 1)
        return res.view(batch_size, n_agents, *self.output_shape)


class RCNN(RecurrentQNetwork):
    """
    Recurrent CNN.
    """

    def __init__(self, input_shape: tuple[int, int, int], extras_size: int, output_shape: tuple[int] | tuple[int, int]):
        super().__init__(output_shape)

        kernel_sizes = [3, 3, 3]
        strides = [1, 1, 1]
        filters = [32, 64, 64]
        self.cnn, self.n_features = make_cnn(input_shape, filters, kernel_sizes, strides)
        self.rnn = RNNQMix(output_shape, self.n_features, extras_size)
        self.extras_shape = (extras_size,)

    @classmethod
    def from_env(cls, env: MARLEnv):
        if env.is_multi_objective:
            output_shape = (env.n_actions, env.reward_space.size)
        else:
            output_shape = (env.n_actions,)
        assert len(env.observation_shape) == 3
        c, h, w = env.observation_shape
        return cls((c, h, w), env.extras_shape[0], output_shape)

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, /, **kwargs) -> torch.Tensor:
        # For transitions, the shape is (batch_size, n_agents, channels, height, width)
        # For episodes, the shape is (time, batch_size, n_agents, channels, height, width)
        *dims, channels, height, width = obs.shape
        obs = obs.view(-1, channels, height, width)
        features = self.cnn.forward(obs)
        features = torch.reshape(features, (*dims, self.n_features))
        extras = extras.view(*dims, *self.extras_shape)
        res = self.rnn.forward(features, extras)
        return res.view(*dims, *self.output_shape)

    def batch_forward(self, obs: torch.Tensor, extras: torch.Tensor, /, **kwargs) -> torch.Tensor:
        *dims, channels, height, width = obs.shape
        obs = obs.reshape(-1, channels, height, width)
        features = self.cnn.forward(obs)
        features = torch.reshape(features, (*dims, self.n_features))
        extras = extras.view(*dims, *self.extras_shape)
        res = self.rnn.batch_forward(features, extras)
        return res.view(*dims, *self.output_shape)


class MAICNetworkRDQN(RecurrentQNetwork, MAIC):
    """
    Source : https://github.com/mansicer/MAIC
    """

    def __init__(self, input_shape: tuple[int, ...], extras_shape: tuple[int, ...], output_size: int, args: "MAICParameters"):
        super().__init__((output_size,))
        self.args = args
        self.n_agents = args.n_agents
        self.latent_dim = args.latent_dim
        self.n_actions = output_size
        self.extras_shape = extras_shape
        if self.args.com:
            NN_HIDDEN_SIZE = args.nn_hidden_size
            self.embed_net = nn.Sequential(
                nn.Linear(args.rnn_hidden_dim, NN_HIDDEN_SIZE),
                nn.BatchNorm1d(NN_HIDDEN_SIZE),
                nn.LeakyReLU(),
                nn.Linear(NN_HIDDEN_SIZE, args.n_agents * args.latent_dim * 2),
            )
            self.inference_net = nn.Sequential(
                nn.Linear(args.rnn_hidden_dim + self.n_actions, NN_HIDDEN_SIZE),
                nn.BatchNorm1d(NN_HIDDEN_SIZE),
                nn.LeakyReLU(),
                nn.Linear(NN_HIDDEN_SIZE, args.latent_dim * 2),
            )
            self.msg_net = nn.Sequential(
                nn.Linear(args.rnn_hidden_dim + args.latent_dim, NN_HIDDEN_SIZE),
                nn.LeakyReLU(),
                nn.Linear(NN_HIDDEN_SIZE, self.n_actions),
            )
            self.w_query = nn.Linear(args.rnn_hidden_dim, args.attention_dim)
            self.w_key = nn.Linear(args.latent_dim, args.attention_dim)

        n_inputs = reduce(operator.mul, input_shape) + extras_shape[0]
        self.fc1 = nn.Linear(n_inputs, args.rnn_hidden_dim)
        self.rnn = nn.GRU(args.rnn_hidden_dim, args.rnn_hidden_dim, batch_first=False)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, self.n_actions)

    @property
    def test_mode(self):
        return not self.training

    def _compute_messages(self, x, bs):
        latent_parameters = self.embed_net(x)
        latent_parameters[:, -self.n_agents * self.latent_dim :] = torch.clamp(
            torch.exp(latent_parameters[:, -self.n_agents * self.latent_dim :]), min=self.args.var_floor
        )
        latent_embed = latent_parameters.reshape(bs * self.n_agents, self.n_agents * self.latent_dim * 2)
        if self.test_mode:
            latent = latent_embed[:, : self.n_agents * self.latent_dim]
        else:
            gaussian_embed = distributions.Normal(
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
            alpha[:, i, i] = -torch.inf
        alpha = F.softmax(alpha, dim=-1).reshape(bs, self.n_agents, self.n_agents, 1)
        if self.test_mode:
            alpha[alpha < (0.25 * 1 / self.n_agents)] = 0
        gated_msg = alpha * msg
        return gated_msg

    def get_values_and_comms(self, obs: torch.Tensor, extras: torch.Tensor) -> tuple:
        *dims, channels, height, width = obs.shape

        is_batch = len(dims) == 3  # episode batch ?
        total_batch = math.prod(dims)
        bs = math.prod(dims[:-1]) if is_batch else 1
        obs = obs.reshape(total_batch, -1)
        if extras is not None:
            extras = extras.reshape(total_batch, *self.extras_shape)
            obs = torch.concat((obs, extras), dim=-1)

        x = F.relu(self.fc1(obs))
        x, self._hidden_states = self.rnn(x, self._hidden_states)
        q = self.fc2(x)

        messages = []
        gated_msg = None
        init_qvalues = q.detach().clone()
        if self.args.com:
            gated_msg = self._compute_messages(x, bs)
            messages = torch.sum(gated_msg, dim=1).view(bs * self.n_agents, self.n_actions)
            q += messages
        return q.view(*dims, *self.output_shape).unsqueeze(-1), gated_msg, messages, init_qvalues

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, /, **kwargs):
        q_values, _, _, _ = self.get_values_and_comms(obs, extras)
        return q_values

    @classmethod
    def from_env(cls, env: MARLEnv[MultiDiscreteSpace], args: "MAICParameters"):  # type: ignore
        return cls(env.observation_shape, env.extras_shape, env.n_actions, args)


class MAICNetworkCNN(MAICNN):
    """
    Source : https://github.com/mansicer/MAIC
    """

    def __init__(self, input_shape: tuple[int, int, int], extras_shape: tuple[int, ...], output_size: int, args: "MAICParameters"):
        assert len(input_shape) == 3, f"CNN can only handle 3D input shapes ({len(input_shape)} here)"
        super().__init__(input_shape, extras_shape, (output_size,))
        self.extras_shape = extras_shape
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
            self.embed_net = nn.Sequential(
                nn.Linear(args.rnn_hidden_dim, NN_HIDDEN_SIZE),
                nn.BatchNorm1d(NN_HIDDEN_SIZE),
                nn.LeakyReLU(),
                nn.Linear(NN_HIDDEN_SIZE, args.n_agents * args.latent_dim * 2),
            )
            self.inference_net = nn.Sequential(
                nn.Linear(args.rnn_hidden_dim + self.n_actions, NN_HIDDEN_SIZE),
                nn.BatchNorm1d(NN_HIDDEN_SIZE),
                nn.LeakyReLU(),
                nn.Linear(NN_HIDDEN_SIZE, args.latent_dim * 2),
            )
            self.msg_net = nn.Sequential(
                nn.Linear(args.rnn_hidden_dim + args.latent_dim, NN_HIDDEN_SIZE),
                nn.LeakyReLU(),
                nn.Linear(NN_HIDDEN_SIZE, self.n_actions),
            )
            self.w_query = nn.Linear(args.rnn_hidden_dim, args.attention_dim)
            self.w_key = nn.Linear(args.latent_dim, args.attention_dim)
        self.cnn, n_features = make_cnn(input_shape, filters, kernel_sizes, strides)
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
            gaussian_embed = distributions.Normal(
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
    def from_env(cls, env: MARLEnv[MultiDiscreteSpace], args: "MAICParameters"):  # type: ignore
        assert len(env.observation_shape) == 3
        c, h, w = env.observation_shape
        return cls((c, h, w), env.extras_shape, env.n_actions, args)
