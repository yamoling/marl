import math
from dataclasses import dataclass

import torch
from marlenv import MultiDiscreteSpace, MARLEnv
from torch.distributions.distribution import Distribution

from marl.models.nn import Critic, ActorCritic, Actor, DiscreteActorCritic, RecurrentNN
from ..utils import make_cnn
from .generic import SimpleRNN, MLP


@dataclass(unsafe_hash=True)
class CNN_ActorCritic(DiscreteActorCritic):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        extras_shape: tuple[int],
        output_shape: tuple[int],
    ):
        super().__init__(output_shape)

        kernel_sizes = [3, 3, 3]
        strides = [1, 1, 1]
        filters = [32, 64, 64]
        self.extras_shape = extras_shape

        self.cnn_actor, n_features = make_cnn(input_shape, filters, kernel_sizes, strides)
        self.cnn_critic, n_features = make_cnn(input_shape, filters, kernel_sizes, strides)
        common_input_size = n_features + extras_shape[0]
        self.actor_nn = torch.nn.Sequential(
            torch.nn.Linear(common_input_size, 128),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(128, *output_shape),
        )
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(common_input_size, 128),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(128, 1),
        )

    def _common_forward(self, data: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        *dims, channels, height, width = data.shape
        leading_dims_size = math.prod(dims)
        data = data.view(leading_dims_size, channels, height, width)
        extras = extras.view(leading_dims_size, *self.extras_shape)
        features = self.cnn_actor.forward(data)
        features = torch.cat((features, extras), dim=-1)
        return features.view(*dims, -1)

    def logits(self, data: torch.Tensor, extras: torch.Tensor, available_actions: torch.Tensor) -> torch.Tensor:
        *dims, channels, height, width = data.shape
        leading_dims_size = math.prod(dims)
        data = data.view(leading_dims_size, channels, height, width)
        extras = extras.view(leading_dims_size, *self.extras_shape)
        features = self.cnn_actor.forward(data)
        features = torch.cat((features, extras), dim=-1)
        features = features.view(*dims, -1)
        logits = self.actor(features)
        return self.mask(logits, available_actions)

    def value(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        *dims, channels, height, width = obs.shape
        leading_dims_size = math.prod(dims)
        obs = obs.view(leading_dims_size, channels, height, width)
        extras = extras.view(leading_dims_size, *self.extras_shape)
        features = self.cnn_critic.forward(obs)
        features = torch.cat((features, extras), dim=-1)
        features = features.view(*dims, -1)
        v = self.critic(features)
        return torch.squeeze(v, dim=-1)

    def policy(self, obs: torch.Tensor, extras: torch.Tensor, available_actions: torch.Tensor) -> Distribution:
        logits = self.logits(obs, extras, available_actions)
        return torch.distributions.Categorical(logits=logits)

    @property
    def value_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.critic.parameters()) + list(self.cnn_critic.parameters())

    @property
    def policy_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.actor_nn.parameters()) + list(self.cnn_actor.parameters())

    @staticmethod
    def from_env(env: MARLEnv[MultiDiscreteSpace]):
        assert len(env.observation_shape) == 3
        assert len(env.extras_shape) == 1
        return CNN_ActorCritic(
            env.observation_shape,
            env.extras_shape,
            (env.n_actions,),
        )


@dataclass(unsafe_hash=True)
class SimpleActorCritic(ActorCritic):
    def __init__(self, input_size: int, extras_size: int, n_actions: int):
        super().__init__(n_actions)
        self.policy_network = torch.nn.Sequential(
            torch.nn.Linear(input_size + extras_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, n_actions),
        )
        self.value_network = torch.nn.Sequential(
            torch.nn.Linear(input_size + extras_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )

    def policy(self, obs: torch.Tensor, extras: torch.Tensor, available_actions: torch.Tensor) -> Distribution:
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

    @classmethod
    def from_env(cls, env: MARLEnv):
        assert len(env.observation_shape) == 1
        assert len(env.extras_shape) == 1
        return SimpleActorCritic(env.observation_shape[0], env.extras_shape[0], env.n_actions)


@dataclass(unsafe_hash=True)
class SimpleRecurrentActorCritic(ActorCritic, RecurrentNN):
    def __init__(self, input_size: int, extras_size: int, n_actions: int):
        ActorCritic.__init__(self, n_actions)
        RecurrentNN.__init__(self, n_actions)
        self.policy_network = SimpleRNN(input_size + extras_size, n_actions)
        self.value_network = SimpleRNN(input_size + extras_size, 1)

    def policy(self, obs: torch.Tensor, extras: torch.Tensor, available_actions: torch.Tensor) -> Distribution:
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

    @classmethod
    def from_env(cls, env: MARLEnv):
        assert len(env.observation_shape) == 1
        assert len(env.extras_shape) == 1
        return SimpleActorCritic(env.observation_shape[0], env.extras_shape[0], env.n_actions)


@dataclass(unsafe_hash=True)
class CNNCritic(Critic):
    def __init__(self, input_shape: tuple[int, ...], n_extras: int):
        assert len(input_shape) == 3, f"CNN can only handle 3D input shapes ({len(input_shape)} here)"
        super().__init__()

        kernel_sizes = [3, 3, 3]
        strides = [1, 1, 1]
        filters = [32, 64, 64]
        assert len(input_shape) == 3
        self.cnn, n_features = make_cnn(input_shape, filters, kernel_sizes, strides)
        self.mlp = MLP(n_features, n_extras, hidden_sizes=[128, 128, 128], output_shape=(1,))
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
class CNNContinuousActor(Actor):
    def __init__(self, input_shape: tuple[int, ...], n_extras: int, action_output_shape: tuple[int, ...]):
        assert len(input_shape) == 3, f"CNN can only handle 3D input shapes ({len(input_shape)} here)"
        super().__init__(action_output_shape)

        kernel_sizes = [3, 3, 3]
        strides = [1, 1, 1]
        filters = [32, 64, 64]
        self.n_actions = math.prod(action_output_shape)
        n_means = self.n_actions
        n_stds = self.n_actions
        self.action_shape = action_output_shape

        assert len(input_shape) == 3
        self.cnn, n_features = make_cnn(input_shape, filters, kernel_sizes, strides)
        self.mlp = MLP(n_features, n_extras, [128, 128, 128], (n_means + n_stds,))

    def _get_distribution(self, means_stds: torch.Tensor):
        batch_size, n_agents, _ = means_stds.shape
        means = means_stds[:, :, : self.n_actions]
        stds = torch.nn.functional.softplus(means_stds[:, :, self.n_actions :])
        # batch_size = means.shape[0] * means.shape[1]
        means = means.view(batch_size, n_agents, *self.action_shape)
        stds = stds.view(batch_size, n_agents, *self.action_shape)
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

    def policy(self, obs: torch.Tensor, extras: torch.Tensor):  # type: ignore
        return self.forward(obs, extras)


@dataclass(unsafe_hash=True)
class CNNContinuousActorCritic(ActorCritic):
    def __init__(self, input_shape: tuple[int, ...], n_extras: int, action_output_shape: tuple[int, ...]):
        assert len(input_shape) == 3, f"CNN can only handle 3D input shapes ({len(input_shape)} here)"
        super().__init__(action_output_shape)

        self.actor_network = CNNContinuousActor(input_shape, n_extras, action_output_shape)
        self.critic = CNNCritic(input_shape, n_extras)

    def value(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        return self.critic.value(obs, extras)

    def policy(self, obs: torch.Tensor, extras: torch.Tensor, available_actions):
        return self.actor_network.forward(obs, extras)

    def forward(self, obs: torch.Tensor, extras: torch.Tensor, available_actions):
        dist = self.actor.forward(obs, extras)
        values = self.critic.forward(obs, extras)
        return dist, values

    @property
    def value_parameters(self):
        return list(self.critic.parameters())

    @property
    def policy_parameters(self):
        return list(self.actor.parameters())


@dataclass(unsafe_hash=True)
class MLPContinuousActorCritic(ActorCritic):
    def __init__(self, input_shape: tuple[int, ...], n_extras: int, action_output_shape: tuple[int, ...]):
        assert len(input_shape) == 1
        super().__init__(action_output_shape)
        self.action_shape = action_output_shape
        self.n_actions = math.prod(action_output_shape)
        n_means = self.n_actions
        n_stds = self.n_actions
        self.actor_nn = torch.nn.Sequential(
            torch.nn.Linear(input_shape[0] + n_extras, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, n_means + n_stds),
        )
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(input_shape[0] + n_extras, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )

    def value(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        obs = torch.concat((obs, extras), dim=-1)
        values = self.critic.forward(obs)
        return torch.squeeze(values, -1)

    def _get_distribution(self, means_stds: torch.Tensor):
        means = means_stds[:, :, : self.n_actions]
        stds = torch.nn.functional.softplus(means_stds[:, :, self.n_actions :])
        *dims, _ = means.shape
        means = means.view(*dims, *self.action_shape)
        stds = stds.view(*dims, *self.action_shape)
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
