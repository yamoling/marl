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
    duelling: bool = True
    mlp_noisy: bool = False

    def __post_init__(self):
        super().__post_init__()
        assert len(self.obs_shape) == 3
        self.cnn, n_features = make_cnn(self.obs_shape, self.filters, self.kernel_sizes, self.strides, "relu")
        n_outputs = self.output_size
        if self.duelling:
            n_outputs += 1
        linears = []
        for _ in range(self.n_agents):
            layers: list[torch.nn.Module] = [torch.nn.Linear(n_features + self.extras_size, self.mlp_sizes[0]), torch.nn.ReLU()]
            for i in range(len(self.mlp_sizes) - 1):
                layers.append(torch.nn.Linear(self.mlp_sizes[i], self.mlp_sizes[i + 1]))
                layers.append(torch.nn.ReLU())
            if self.mlp_noisy:
                layers.append(NoisyLinear(self.mlp_sizes[-1], n_outputs))
            else:
                layers.append(torch.nn.Linear(self.mlp_sizes[-1], n_outputs))
            linears.append(torch.nn.Sequential(*layers))
        self.linears = torch.nn.ModuleList(linears)

    @classmethod
    def from_env(
        cls,
        env: DiscreteMARLEnv | EnvConfig[DiscreteMARLEnv],
        mlp_sizes: tuple[int, ...] = (64, 64),
        duelling: bool = True,
        mlp_noisy: bool = False,
        **kwargs,
    ):
        assert len(env.observation_shape) == 3
        c, h, w = env.observation_shape
        return cls(
            env.n_actions,
            (c, h, w),
            env.extras_shape,
            env.n_agents,
            mlp_sizes=mlp_sizes,
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

    def __hash__(self):
        return id(self)
