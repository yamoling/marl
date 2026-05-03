import math
from dataclasses import dataclass
from typing import Sequence

import torch
from marlenv import MARLEnv
from torch import Tensor

from marl.models.nn import ActivationType, Actor, ActorCritic, Critic, DiscreteActorCritic, RecurrentNN, get_activation

from ..utils import make_cnn
from .generic import CNN, MLP, RNN


@dataclass(unsafe_hash=True)
class CNN_ActorCritic(DiscreteActorCritic):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        extras_size: int,
        n_actions: int,
        activation: ActivationType,
        mlp_sizes: Sequence[int],
    ):
        super().__init__((n_actions,))

        kernel_sizes = [3, 3, 3]
        strides = [1, 1, 1]
        filters = [32, 64, 64]
        self.extras_size = extras_size

        self.cnn_actor, n_features = make_cnn(input_shape, filters, kernel_sizes, strides, "relu")
        self.cnn_critic, n_features = make_cnn(input_shape, filters, kernel_sizes, strides, "relu")
        layer_sizes = [n_features + extras_size, *mlp_sizes]
        self.actor_nn = torch.nn.Sequential()
        self.critic = torch.nn.Sequential()
        for size, next_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.actor_nn.append(torch.nn.Linear(size, next_size))
            self.critic.append(torch.nn.Linear(size, next_size))
            self.actor_nn.append(get_activation(activation))
            self.critic.append(get_activation(activation))
        self.actor_nn.append(torch.nn.Linear(layer_sizes[-1], n_actions))
        self.critic.append(torch.nn.Linear(layer_sizes[-1], 1))

    def _common_forward(self, data: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        *dims, channels, height, width = data.shape
        leading_dims_size = math.prod(dims)
        data = data.view(leading_dims_size, channels, height, width)
        extras = extras.view(leading_dims_size, self.extras_size)
        features = self.cnn_actor.forward(data)
        features = torch.cat((features, extras), dim=-1)
        return features.view(*dims, -1)

    def logits(self, obs: torch.Tensor, extras: torch.Tensor, available_actions: torch.Tensor):
        *dims, channels, height, width = obs.shape
        leading_dims_size = math.prod(dims)
        obs = obs.view(leading_dims_size, channels, height, width)
        extras = extras.view(leading_dims_size, self.extras_size)
        features = self.cnn_actor.forward(obs)
        features = torch.cat((features, extras), dim=-1)
        features = features.view(*dims, -1)
        logits = self.actor_nn(features)
        return self.mask(logits, available_actions)

    def value(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        *dims, channels, height, width = obs.shape
        leading_dims_size = math.prod(dims)
        obs = obs.view(leading_dims_size, channels, height, width)
        extras = extras.view(leading_dims_size, self.extras_size)
        features = self.cnn_critic.forward(obs)
        features = torch.cat((features, extras), dim=-1)
        features = features.view(*dims, -1)
        v = self.critic(features)
        return torch.squeeze(v, dim=-1)

    def policy(self, obs: torch.Tensor, extras: torch.Tensor, available_actions: torch.Tensor):
        logits = self.logits(obs, extras, available_actions)
        return torch.distributions.Categorical(logits=logits)

    @property
    def value_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.critic.parameters()) + list(self.cnn_critic.parameters())

    @property
    def policy_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.actor_nn.parameters()) + list(self.cnn_actor.parameters())


@dataclass(unsafe_hash=True)
class SimpleActorCritic(ActorCritic[torch.distributions.Categorical]):
    def __init__(
        self,
        input_size: int,
        extras_size: int,
        n_actions: int,
        mlp_sizes: Sequence[int],
        activation: ActivationType,
    ):
        super().__init__((n_actions,))
        layer_sizes = [input_size + extras_size, *mlp_sizes]
        self.policy_network = torch.nn.Sequential()
        self.value_network = torch.nn.Sequential()
        for prev_size, next_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.policy_network.append(torch.nn.Linear(prev_size, next_size))
            self.policy_network.append(get_activation(activation))
            self.value_network.append(torch.nn.Linear(prev_size, next_size))
            self.value_network.append(get_activation(activation))
        self.policy_network.append(torch.nn.Linear(mlp_sizes[-1], n_actions))
        self.value_network.append(torch.nn.Linear(mlp_sizes[-1], 1))

    def policy(self, obs: torch.Tensor, extras: torch.Tensor, available_actions: torch.Tensor):
        x = torch.cat((obs, extras), dim=-1)
        logits = self.policy_network(x)
        logits = self.mask(logits, available_actions, replacement=-torch.inf)
        return torch.distributions.Categorical(logits=logits)

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, available_actions: torch.Tensor):
        return self.policy(obs, extras, available_actions), self.value(obs, extras)

    def value(self, obs: torch.Tensor, extras: torch.Tensor):
        obs = torch.cat((obs, extras), dim=-1)
        return torch.squeeze(self.value_network(obs), -1)

    @property
    def value_parameters(self):
        return list(self.value_network.parameters())

    @property
    def policy_parameters(self):
        return list(self.policy_network.parameters())


class SimpleRecurrentActorCritic(ActorCritic[torch.distributions.Categorical], RecurrentNN):
    def __init__(self, n_actions: int, obs_size: int, extras_size: int, mlp_sizes: Sequence[int], activation: ActivationType):
        ActorCritic.__init__(self, (n_actions,))
        RecurrentNN.__init__(self, (n_actions,))
        self.policy_network = RNN(self.output_shape, obs_size, extras_size, mlp_sizes, activation)
        self.value_network = RNN((1,), obs_size, extras_size, mlp_sizes, activation)

    def policy(self, obs: torch.Tensor, extras: torch.Tensor, available_actions: torch.Tensor):
        logits = self.policy_network.forward(obs, extras)
        logits = self.mask(logits, available_actions, replacement=-torch.inf)
        return torch.distributions.Categorical(logits=logits)

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, available_actions: torch.Tensor):
        return self.policy(obs, extras, available_actions), self.value(obs, extras)

    def value(self, obs: torch.Tensor, extras: torch.Tensor):
        obs = torch.cat((obs, extras), dim=-1)
        return torch.squeeze(self.value_network.forward(obs, extras), -1)

    @property
    def value_parameters(self):
        return list(self.value_network.parameters())

    @property
    def policy_parameters(self):
        return list(self.policy_network.parameters())


@dataclass(unsafe_hash=True)
class CNNCritic(Critic):
    def __init__(self, input_shape: tuple[int, int, int], n_extras: int, activation: ActivationType, hidden_sizes: Sequence[int]):
        assert len(input_shape) == 3, f"CNN can only handle 3D input shapes ({len(input_shape)} here)"
        super().__init__((1,))

        kernel_sizes = [3, 3, 3]
        strides = [1, 1, 1]
        filters = [32, 64, 64]
        self.cnn, n_features = make_cnn(input_shape, filters, kernel_sizes, strides, activation)
        self.mlp = MLP((1,), n_features, n_extras, hidden_sizes, hidden_activation=activation)
        self.n_extras = n_extras

    def value(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        return self.forward(obs, extras)

    def forward(
        self,
        data: torch.Tensor,
        extras: torch.Tensor,
    ):
        *dims, channels, height, width = data.shape
        leading_dims_size = math.prod(dims)
        data = data.reshape(leading_dims_size, channels, height, width)
        extras = extras.reshape(leading_dims_size, self.n_extras)
        features = self.cnn.forward(data)
        value = self.mlp.forward(features, extras)
        value = value.reshape(*dims)
        return value


@dataclass(unsafe_hash=True)
class CNNDiscreteAC(ActorCritic[torch.distributions.Categorical]):
    def __init__(self, input_shape: tuple[int, int, int], n_extras: int, n_actions: int):
        super().__init__((n_actions,))
        self.actor = CNN((n_actions,), input_shape, n_extras, [128, 128], "relu")
        self.critic = CNN((1,), input_shape, n_extras, [128, 128], "relu")

    def policy(self, obs: torch.Tensor, extras: torch.Tensor, available_actions: torch.Tensor):
        logits = self.actor.forward(obs, extras)
        logits[~available_actions] = -torch.inf
        return torch.distributions.Categorical(logits=logits)

    def value(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        values = self.critic.forward(obs, extras)
        return values.squeeze(-1)

    @property
    def policy_parameters(self):
        return list(self.actor.parameters())

    @property
    def value_parameters(self):
        return list(self.critic.parameters())

    @classmethod
    def from_env(cls, env: MARLEnv):
        assert len(env.observation_shape) == 3
        return cls(env.observation_shape, env.extras_size, env.n_actions)

    def to(self, device: torch.device):
        self.actor.to(device)
        self.critic.to(device)
        return self


@dataclass(unsafe_hash=True)
class CNNContinuousActor(Actor):
    def __init__(
        self, input_shape: tuple[int, int, int], extras_size: int, n_actions: int, activation: ActivationType, hidden_sizes: Sequence[int]
    ):
        super().__init__((n_actions,))

        kernel_sizes = [3, 3, 3]
        strides = [1, 1, 1]
        filters = [32, 64, 64]
        self.n_actions = n_actions
        c, h, w = input_shape
        self.cnn, n_features = make_cnn((c, h, w), filters, kernel_sizes, strides, activation)
        self.mlp = MLP((2 * n_actions,), n_features, extras_size, hidden_sizes, activation)

    def _get_distribution(self, means_stds: torch.Tensor):
        batch_size, n_agents, _ = means_stds.shape
        means = means_stds[:, :, : self.n_actions]
        stds = torch.nn.functional.softplus(means_stds[:, :, self.n_actions :])
        # batch_size = means.shape[0] * means.shape[1]
        means = means.view(batch_size, n_agents, self.n_actions)
        stds = stds.view(batch_size, n_agents, self.n_actions)
        return torch.distributions.Normal(means, stds)

    def forward(
        self,
        data: torch.Tensor,
        extras: torch.Tensor,
    ):
        *dims, channels, height, width = data.shape
        leading_dims_size = math.prod(dims)
        data = data.view(leading_dims_size, channels, height, width)
        features = self.cnn.forward(data)
        features = features.view(*dims, -1)
        means_stds = self.mlp.forward(features, extras)
        dist = self._get_distribution(means_stds)
        return dist

    def policy(self, obs: torch.Tensor, extras: torch.Tensor, available_actions):
        return self.forward(obs, extras)


@dataclass(unsafe_hash=True)
class CNNContinuousActorCritic(ActorCritic[torch.distributions.Normal]):
    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        extras_size: int,
        n_actions: int,
        hidden_sizes: Sequence[int],
        activation: ActivationType,
    ):
        super().__init__((n_actions,))

        self.actor_network = CNNContinuousActor(obs_shape, extras_size, n_actions, activation, hidden_sizes)
        self.critic = CNNCritic(obs_shape, extras_size, activation, hidden_sizes)

    def value(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        return self.critic.value(obs, extras)

    def policy(self, obs: torch.Tensor, extras: torch.Tensor, available_actions):
        return self.actor_network.forward(obs, extras)

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, available_actions: torch.Tensor | None):
        dist = self.actor_network.forward(obs, extras)
        values = self.critic.forward(obs, extras)
        return dist, values

    @property
    def value_parameters(self):
        return list(self.critic.parameters())

    @property
    def policy_parameters(self):
        return list(self.actor_network.parameters())


@dataclass(unsafe_hash=True)
class MLPContinuousActorCritic(ActorCritic):
    def __init__(
        self,
        obs_size: int,
        extras_size: int,
        n_actions: int,
        hidden_sizes: Sequence[int],
        hidden_activation: ActivationType,
    ):
        super().__init__((n_actions,))
        self.n_actions = n_actions
        layer_sizes = [obs_size + extras_size, *hidden_sizes]
        self.actor_nn = torch.nn.Sequential()
        self.critic = torch.nn.Sequential()
        for size, next_size in zip(layer_sizes, layer_sizes[1:]):
            self.actor_nn.append(torch.nn.Linear(size, next_size))
            self.critic.append(torch.nn.Linear(size, next_size))
            self.actor_nn.append(get_activation(hidden_activation))
            self.critic.append(get_activation(hidden_activation))
        self.actor_nn.append(torch.nn.Linear(layer_sizes[-1], 2 * n_actions))  # means and stds
        self.critic.append(torch.nn.Linear(layer_sizes[-1], 1))

    def value(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        obs = torch.concat((obs, extras), dim=-1)
        values = self.critic.forward(obs)
        return torch.squeeze(values, -1)

    def _get_distribution(self, means_stds: torch.Tensor):
        means = means_stds[:, :, : self.n_actions]
        stds = torch.nn.functional.softplus(means_stds[:, :, self.n_actions :])
        *dims, _ = means.shape
        means = means.view(*dims, self.n_actions)
        stds = stds.view(*dims, self.n_actions)
        return torch.distributions.Normal(means, stds)

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, available_actions):
        x = torch.concat((obs, extras), dim=-1)
        values = self.critic.forward(x)
        means_stds = self.actor_nn.forward(x)
        dist = self._get_distribution(means_stds)
        return dist, torch.squeeze(values, -1)

    def policy(self, obs: torch.Tensor, extras: torch.Tensor, available_actions):
        x = torch.concat((obs, extras), dim=-1)
        means_and_stds = self.actor_nn.forward(x)
        return self._get_distribution(means_and_stds)

    @property
    def value_parameters(self):
        return list(self.critic.parameters())

    @property
    def policy_parameters(self):
        return list(self.actor_nn.parameters())


@dataclass
class CNNActor(Actor, CNN):
    def __init__(self, obs_shape: tuple[int, ...], extras_size: int, n_actions: int):
        assert len(obs_shape) == 3
        (c, h, w) = obs_shape
        Actor.__init__(self, (n_actions,))
        CNN.__init__(self, (n_actions,), (c, h, w), extras_size, [128, 128], "relu")

    def policy(self, obs: torch.Tensor, extras: torch.Tensor, available_actions: torch.Tensor) -> torch.distributions.Distribution:
        logits = CNN.forward(self, obs, extras)
        logits[~available_actions] = -torch.inf
        return torch.distributions.Categorical(logits=logits)

    def forward(self, obs: Tensor, extras: Tensor, *args, **kwargs):
        return CNN.forward(self, obs, extras)

    def __hash__(self):
        return hash(self.name)
