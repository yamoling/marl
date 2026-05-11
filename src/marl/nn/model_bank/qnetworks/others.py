from dataclasses import KW_ONLY, dataclass

import torch
from marlenv import DiscreteMARLEnv

from marl.env import EnvConfig
from marl.models.nn import QNetwork
from marl.nn.layers import NoisyLinear
from marl.nn.utils import make_cnn


@dataclass
class IndependentCNN(QNetwork):
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
