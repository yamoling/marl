from dataclasses import KW_ONLY, dataclass
from typing import Sequence, cast

import torch
from marlenv import DiscreteMARLEnv

from marl.env import EnvConfig
from marl.models.nn import ActivationType, QNetwork, RecurrentQNetwork
from marl.nn.layers import NoisyLinear
from marl.nn.model_bank.generic import CRNN, RNN
from marl.nn.utils import make_cnn


def _stack_agent_outputs(
    outputs: list[torch.Tensor],
    n_agents: int,
    output_shape: tuple[int, ...],
    stack_dim: int,
    dims: tuple[int, ...],
) -> torch.Tensor:
    res = torch.stack(outputs, dim=stack_dim)
    return res.view(*dims, n_agents, *output_shape)


@dataclass
class IQCNN(QNetwork):
    """
    CNN whose flattened output is concatenated with the extras to be fed to the linear layers.

    The CNN part of the network is shared but the linear layers are separated.
    """

    n_agents: int
    _: KW_ONLY
    mlp_sizes: tuple[int, ...] = (256, 128)
    kernel_sizes: tuple[int, ...] = (3, 3, 3)
    strides: tuple[int, ...] = (1, 1, 1)
    filters: tuple[int, ...] = (32, 64, 64)

    def __post_init__(self):
        super().__post_init__()
        assert len(self.obs_shape) == 3
        self.cnn, n_features = make_cnn(self.obs_shape, self.filters, self.kernel_sizes, self.strides, "relu")
        linears = []
        for _ in range(self.n_agents):
            layers: list[torch.nn.Module] = [torch.nn.Linear(n_features + self.extras_size, self.mlp_sizes[0]), torch.nn.ReLU()]
            for i in range(len(self.mlp_sizes) - 1):
                layers.append(torch.nn.Linear(self.mlp_sizes[i], self.mlp_sizes[i + 1]))
                layers.append(torch.nn.ReLU())
            if self.noisy:
                layers.append(NoisyLinear(self.mlp_sizes[-1], self.output_size))
            else:
                layers.append(torch.nn.Linear(self.mlp_sizes[-1], self.output_size))
            linears.append(torch.nn.Sequential(*layers))
        self.linears = torch.nn.ModuleList(linears)

    @classmethod
    def from_env(
        cls,
        env: DiscreteMARLEnv | EnvConfig[DiscreteMARLEnv],
        mlp_sizes: tuple[int, ...] = (256, 128),
        duelling: bool = True,
        noisy: bool = False,
        **kwargs,
    ):
        assert len(env.observation_shape) == 3
        return cls(
            env.n_actions,
            env.observation_shape,
            env.extras_shape,
            env.n_agents,
            mlp_sizes=mlp_sizes,
            duelling=duelling,
            noisy=noisy,
            **kwargs,
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
        # The shape is (batch_size, n_agents, ...), therefore stack on the agent dimension (i.e. dim=1).
        res = torch.stack([self.linears[i].forward(features[:, i]) for i in range(self.n_agents)], dim=1)
        return res.view(batch_size, n_agents, *self.output_shape)

    def __hash__(self):
        return id(self)


@dataclass
class IQMLP(QNetwork):
    """Independent per-agent MLPs."""

    n_agents: int
    _: KW_ONLY
    mlp_sizes: Sequence[int] = (256, 128)

    def __post_init__(self):
        super().__post_init__()
        linears = []
        for _ in range(self.n_agents):
            layers: list[torch.nn.Module] = [torch.nn.Linear(self.obs_size + self.extras_size, self.mlp_sizes[0]), torch.nn.ReLU()]
            for i in range(len(self.mlp_sizes) - 1):
                layers.append(torch.nn.Linear(self.mlp_sizes[i], self.mlp_sizes[i + 1]))
                layers.append(torch.nn.ReLU())
            if self.noisy:
                layers.append(NoisyLinear(self.mlp_sizes[-1], self.output_size))
            else:
                layers.append(torch.nn.Linear(self.mlp_sizes[-1], self.output_size))
            linears.append(torch.nn.Sequential(*layers))
        self.linears = torch.nn.ModuleList(linears)

    @classmethod
    def from_env(
        cls,
        env: DiscreteMARLEnv | EnvConfig[DiscreteMARLEnv],
        mlp_sizes: Sequence[int] = (256, 128),
        duelling: bool = True,
        noisy: bool = False,
        **kwargs,
    ):
        assert len(env.observation_shape) == 1
        return cls(
            env.n_actions,
            env.observation_shape,
            env.extras_shape,
            env.n_agents,
            mlp_sizes=mlp_sizes,
            duelling=duelling,
            noisy=noisy,
            **kwargs,
        )

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, /, **kwargs) -> torch.Tensor:
        *dims, n_agents, obs_size = obs.shape
        obs = obs.reshape(-1, n_agents, obs_size)
        extras = extras.reshape(-1, n_agents, self.extras_size)
        outputs = [self.linears[i].forward(torch.concatenate((obs[:, i], extras[:, i]), dim=-1)) for i in range(self.n_agents)]
        return _stack_agent_outputs(outputs, n_agents, self.output_shape, len(dims), tuple(dims))

    def __hash__(self):
        return id(self)


@dataclass
class IQRNN(RecurrentQNetwork):
    """Independent per-agent recurrent Q-networks."""

    n_agents: int
    _: KW_ONLY
    mlp_head_sizes: Sequence[int] = (256,)
    mlp_tail_sizes: Sequence[int] = (128,)
    hidden_activation: ActivationType = "relu"

    def __post_init__(self):
        super().__post_init__()
        self.rnns = torch.nn.ModuleList(
            [
                RNN(
                    self.output_shape,
                    self.obs_size,
                    self.extras_size,
                    mlp_head_sizes=self.mlp_head_sizes,
                    mlp_tail_sizes=self.mlp_tail_sizes,
                    hidden_activation=self.hidden_activation,
                )
                for _ in range(self.n_agents)
            ]
        )

    @classmethod
    def from_env(
        cls,
        env: DiscreteMARLEnv | EnvConfig[DiscreteMARLEnv],
        mlp_head_sizes: Sequence[int] = (256,),
        mlp_tail_sizes: Sequence[int] = (128,),
        hidden_activation: ActivationType = "relu",
        duelling: bool = True,
        noisy: bool = False,
        **kwargs,
    ):
        assert len(env.observation_shape) == 1
        return cls(
            env.n_actions,
            env.observation_shape,
            env.extras_shape,
            env.n_agents,
            mlp_head_sizes=mlp_head_sizes,
            mlp_tail_sizes=mlp_tail_sizes,
            hidden_activation=hidden_activation,
            duelling=duelling,
            noisy=noisy,
            **kwargs,
        )

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, /, **kwargs) -> torch.Tensor:
        *dims, n_agents, _ = obs.shape
        outputs = [self.rnns[i].forward(obs.select(-2, i), extras.select(-2, i), **kwargs) for i in range(self.n_agents)]
        return _stack_agent_outputs(outputs, n_agents, self.output_shape, len(dims), tuple(dims))

    def reset_hidden_states(self):
        super().reset_hidden_states()
        for i in range(self.n_agents):
            cast(RNN, self.rnns[i]).reset_hidden_states()

    def __hash__(self):
        return id(self)


@dataclass
class ICRNN(RecurrentQNetwork):
    """Independent per-agent convolutional recurrent Q-networks."""

    n_agents: int
    _: KW_ONLY
    mlp_head_sizes: Sequence[int] = (256,)
    mlp_tail_sizes: Sequence[int] = (128,)
    hidden_activation: ActivationType = "relu"
    kernel_sizes: Sequence[int] = (3, 3, 3)
    strides: Sequence[int] = (1, 1, 1)
    filters: Sequence[int] = (32, 64, 64)

    def __post_init__(self):
        super().__post_init__()
        assert len(self.obs_shape) == 3
        self.crnns = torch.nn.ModuleList(
            [
                CRNN(
                    self.output_shape,
                    self.obs_shape,
                    self.extras_size,
                    hidden_activation=self.hidden_activation,
                    mlp_head_sizes=self.mlp_head_sizes,
                    mlp_tail_sizes=self.mlp_tail_sizes,
                    kernel_sizes=self.kernel_sizes,
                    strides=self.strides,
                    filters=self.filters,
                )
                for _ in range(self.n_agents)
            ]
        )

    @classmethod
    def from_env(
        cls,
        env: DiscreteMARLEnv | EnvConfig[DiscreteMARLEnv],
        mlp_head_sizes: Sequence[int] = (256,),
        mlp_tail_sizes: Sequence[int] = (128,),
        hidden_activation: ActivationType = "relu",
        duelling: bool = True,
        noisy: bool = False,
        **kwargs,
    ):
        assert len(env.observation_shape) == 3
        return cls(
            env.n_actions,
            env.observation_shape,
            env.extras_shape,
            env.n_agents,
            mlp_head_sizes=mlp_head_sizes,
            mlp_tail_sizes=mlp_tail_sizes,
            hidden_activation=hidden_activation,
            duelling=duelling,
            noisy=noisy,
            **kwargs,
        )

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, /, **kwargs) -> torch.Tensor:
        *dims, n_agents, _, _, _ = obs.shape
        outputs = [self.crnns[i].forward(obs.select(-4, i), extras.select(-4, i), **kwargs) for i in range(self.n_agents)]
        return _stack_agent_outputs(outputs, n_agents, self.output_shape, len(dims), tuple(dims))

    def reset_hidden_states(self):
        super().reset_hidden_states()
        for i in range(self.n_agents):
            cast(CRNN, self.crnns[i]).reset_hidden_states()

    def __hash__(self):
        return id(self)


IQCNN = IQCNN
