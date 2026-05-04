import math
from abc import abstractmethod
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING, Literal, Sequence

import torch
from marlenv import MARLEnv, MultiDiscreteSpace
from torch import Tensor

from marl.models.nn import NN, ActivationType, QNetwork, RecurrentQNetwork

from ..layers import NoisyLinear
from ..utils import make_cnn
from .generic import CNN, CRNN, MLP, RNN

if TYPE_CHECKING:
    from marl.config import EnvConfig


@dataclass(unsafe_hash=True)
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
            cnn_output_activation = self.hidden_activation
        else:
            self.noisy_layer = None
            cnn_output_shape = self.output_shape
            cnn_output_activation = None
        self.cnn = CNN(
            cnn_output_shape,
            self.obs_shape,
            self.extras_size,
            self.mlp_sizes,
            self.hidden_activation,
            output_activation=cnn_output_activation,
        )

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, /, **kwargs):
        x = self.cnn.forward(obs, extras, **kwargs)
        if self.noisy_layer is not None:
            return self.noisy_layer.forward(x)
        return x


@dataclass(unsafe_hash=True)
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
                self.hidden_sizes,
                self.activation,
                output_activation=self.activation,
            )
            self.noisy_layer = NoisyLinear(self.hidden_sizes[1], self.n_actions)
        else:
            self.nn = MLP(self.output_shape, self.obs_size, self.extras_size, self.hidden_sizes, self.activation)
            self.noisy_layer = None

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, /, **kwargs):
        x = self.nn.forward(obs, extras, **kwargs)
        if self.noisy_layer is not None:
            return self.noisy_layer.forward(x)
        return x


@dataclass(unsafe_hash=True)
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


@dataclass(unsafe_hash=True)
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


def from_env(env: MARLEnv[MultiDiscreteSpace] | EnvConfig, recurrent: bool = False) -> QNetwork:
    if not isinstance(env, MARLEnv):
        env = env.make()
    match (len(env.observation_shape), recurrent):
        case (1, False):
            return QMLP.from_env(env)
        case (3, False):
            return QCNN.from_env(env)
        case (1, True):
            return QRNN.from_env(env)
        case (3, True):
            return QCRNN.from_env(env)
    raise NotImplementedError("Unsupported environment observation shape and recurrent combination. Create your own Q-network !")


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
    noise_size: int
    n_agents: int
    _: KW_ONLY
    head: NN = field(init=False)
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
            hidden_activation="relu",
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
        self.head = MLP(
            (self.agent_output_size,),
            self.obs_size,
            self.extras_size - self.noise_size,
            (256, 256),
            "relu",
            noisy=False,
        )

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
        self.cnn, n_features = make_cnn(input_shape, filters, kernel_sizes, strides, "relu")
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
