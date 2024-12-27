import math
from dataclasses import dataclass

import torch
from marlenv import DiscreteActionSpace, MARLEnv

from marl.models.nn import ContinuousActorCriticNN, DiscreteActorCriticNN

from ..utils import make_cnn


class DDPG_NN_TEST(DiscreteActorCriticNN):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        extras_shape: tuple[int],
        output_shape: tuple[int],
        n_agents: int,
        n_actions: int,
        state_size: int,
    ):
        assert len(input_shape) == 3, f"CNN can only handle 3D input shapes ({len(input_shape)} here)"
        super().__init__(input_shape, extras_shape, output_shape[0])

        kernel_sizes = [3, 3, 3]
        strides = [1, 1, 1]
        filters = [32, 64, 64]

        self.cnn, n_features = make_cnn(self.input_shape, filters, kernel_sizes, strides)
        input_size = n_features + self.extras_shape[0]
        self.policy_network = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, *output_shape),
        )

        self.value_network = torch.nn.Sequential(
            # torch.nn.Linear(state_size + n_actions * n_agents, 128),
            torch.nn.Linear(n_features + n_actions * n_agents, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )

    def _cnn_forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Check that the input has the correct shape (at least 4 dimensions)
        *dims, channels, height, width = obs.shape
        leading_dims_size = math.prod(dims)
        obs = obs.view(leading_dims_size, channels, height, width)
        features = self.cnn.forward(obs)
        return features

    def forward(self, obs: torch.Tensor, extras: torch.Tensor):
        features = self._cnn_forward(obs)
        extras = extras.view(-1, *self.extras_shape)
        features = torch.cat((features, extras), dim=-1)
        return self.policy(features), 0

    def policy(self, obs: torch.Tensor):
        logits = self.policy_network(obs)
        # clipped_logits = torch.clamp(logits, 0, 4)
        return logits

    def value(self, state: torch.Tensor, extras: torch.Tensor, actions: torch.Tensor):
        actions = actions.view(actions.shape[0], -1)
        # features = torch.cat((state, actions), dim=1)
        features = self._cnn_forward(state)
        # print(features.shape)
        # print(actions.shape)
        features = torch.cat((features, actions), dim=1)
        # print(features.shape)
        return self.value_network(features).squeeze()

    @classmethod
    def from_env(cls, env: MARLEnv):
        assert len(env.observation_shape) == 3
        assert len(env.extra_shape) == 1
        return cls(
            input_shape=env.observation_shape,
            extras_shape=env.extra_shape,
            output_shape=(env.n_actions,),
            n_agents=env.n_agents,
            n_actions=env.n_actions,
            state_size=env.state_shape[0],
        )

    @property
    def value_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.value_network.parameters())

    @property
    def policy_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.cnn.parameters()) + list(self.policy_network.parameters())


@dataclass
class CNN_ActorCritic(DiscreteActorCriticNN):
    def __init__(self, input_shape: tuple[int, int, int], extras_shape: tuple[int], output_shape: tuple[int]):
        assert len(input_shape) == 3, f"CNN can only handle 3D input shapes ({len(input_shape)} here)"
        super().__init__(input_shape, extras_shape, output_shape[0])

        kernel_sizes = [3, 3, 3]
        strides = [1, 1, 1]
        filters = [32, 64, 64]

        self.cnn, n_features = make_cnn(self.input_shape, filters, kernel_sizes, strides)
        common_input_size = n_features + self.extras_shape[0]
        self.common = torch.nn.Sequential(
            torch.nn.Linear(common_input_size, 128),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(0.1),
        )

        self.policy_network = torch.nn.Linear(128, *output_shape)
        self.value_network = torch.nn.Linear(128, 1)

    def _common_forward(self, data: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        *dims, channels, height, width = data.shape
        leading_dims_size = math.prod(dims)
        data = data.view(leading_dims_size, channels, height, width)
        features = self.cnn.forward(data)
        features = torch.cat((features, extras), dim=-1)
        return self.common(features)

    def logits(self, data: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        x = self._common_forward(data, extras)
        logits = self.policy_network(x)
        return logits

    def value(self, data: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        x = self._common_forward(data, extras)
        return self.value_network(x)

    def forward(
        self,
        data: torch.Tensor,
        extras: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._common_forward(data, extras)
        logits = self.policy_network(x)
        if action_mask is not None:
            logits = logits * action_mask
        pi = torch.nn.functional.softmax(logits, dim=-1)
        v = self.value_network(x)
        return pi, v

    @property
    def value_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.cnn.parameters()) + list(self.common.parameters()) + list(self.value_network.parameters())

    @property
    def policy_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.cnn.parameters()) + list(self.common.parameters()) + list(self.policy_network.parameters())


@dataclass
class SimpleActorCritic(DiscreteActorCriticNN):
    def __init__(self, input_size: int, extras_size: int, n_actions: int):
        super().__init__((input_size,), (extras_size,), n_actions)
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

    def logits(self, data: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        x = torch.cat((data, extras), dim=-1)
        return self.policy_network(x)

    def forward(self, obs: torch.Tensor, extras: torch.Tensor):
        x = torch.cat((obs, extras), dim=-1)
        return self.policy_network(x), self.value_network(x)

    def value(self, x: torch.Tensor, extras: torch.Tensor):
        x = torch.cat((x, extras), dim=-1)
        return self.value_network(x)

    @property
    def value_parameters(self):
        return list(self.common.parameters()) + list(self.value_network.parameters())

    @property
    def policy_parameters(self):
        return list(self.common.parameters()) + list(self.policy_network.parameters())

    @classmethod
    def from_env[A](cls, env: MARLEnv[A, DiscreteActionSpace]):
        assert len(env.observation_shape) == 1
        assert len(env.extra_shape) == 1
        return SimpleActorCritic(env.observation_shape[0], env.extra_shape[0], env.n_actions)


@dataclass
class CNNContinuousActorCritic(ContinuousActorCriticNN):
    def __init__(self, input_shape: tuple[int, ...], n_extras: int, action_output_shape: tuple[int, ...]):
        assert len(input_shape) == 3, f"CNN can only handle 3D input shapes ({len(input_shape)} here)"
        super().__init__(input_shape, (n_extras,), action_output_shape)

        kernel_sizes = [3, 3, 3]
        strides = [1, 1, 1]
        filters = [32, 64, 64]
        self.cnn, n_features = make_cnn(self.input_shape, filters, kernel_sizes, strides)
        common_input_size = n_features + n_extras
        self.common = torch.nn.Sequential(
            torch.nn.Linear(common_input_size, 128),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(0.1),
        )

        self.n_actions = math.prod(action_output_shape)
        n_means = self.n_actions
        n_stds = self.n_actions
        self.policy_network = torch.nn.Linear(128, n_means + n_stds)
        self.value_network = torch.nn.Linear(128, 1)

    def _common_forward(self, data: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        if len(data.shape) != 4:
            *dims, channels, height, width = data.shape
            leading_dims_size = math.prod(dims)
            data = data.view(leading_dims_size, channels, height, width)
        features = self.cnn.forward(data)
        features = torch.cat((features, extras), dim=-1)
        return self.common(features)

    def value(self, data: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        x = self._common_forward(data, extras)
        return self.value_network(x)

    def _get_distribution(self, means_stds: torch.Tensor):
        means = means_stds[:, : self.n_actions]
        stds = torch.nn.functional.softplus(means_stds[:, self.n_actions :])
        batch_size = means.shape[0]
        means = means.view(batch_size, *self.action_output_shape)
        stds = stds.view(batch_size, *self.action_output_shape)
        return torch.distributions.Normal(means, stds)

    def forward(
        self,
        data: torch.Tensor,
        extras: torch.Tensor,
    ):
        x = self._common_forward(data, extras)
        values = self.value_network.forward(x)
        means_stds = self.policy_network.forward(x)
        dist = self._get_distribution(means_stds)
        return dist, values

    def policy(self, obs: torch.Tensor, extras: torch.Tensor):
        x = self._common_forward(obs, extras)
        means_and_stds = self.policy_network.forward(x)
        return self._get_distribution(means_and_stds)

    @property
    def value_parameters(self):
        return list(self.cnn.parameters()) + list(self.common.parameters()) + list(self.value_network.parameters())

    @property
    def policy_parameters(self):
        return list(self.cnn.parameters()) + list(self.common.parameters()) + list(self.policy_network.parameters())


@dataclass(unsafe_hash=True)
class MLPContinuousActorCritic(ContinuousActorCriticNN):
    def __init__(self, input_shape: tuple[int, ...], n_extras: int, action_output_shape: tuple[int, ...]):
        assert len(input_shape) == 1
        super().__init__(input_shape, (n_extras,), action_output_shape)

        self.n_actions = math.prod(action_output_shape)
        n_means = self.n_actions
        n_stds = self.n_actions
        self.actor = torch.nn.Sequential(
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

    def value(self, data: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        data = torch.concat((data, extras), dim=-1)
        return self.critic.forward(data)

    def _get_distribution(self, means_stds: torch.Tensor):
        means = means_stds[:, :, : self.n_actions]
        stds = torch.nn.functional.softplus(means_stds[:, :, self.n_actions :])
        *dims, _ = means.shape
        means = means.view(*dims, *self.action_output_shape)
        stds = stds.view(*dims, *self.action_output_shape)
        return torch.distributions.Normal(means, stds)

    def forward(
        self,
        data: torch.Tensor,
        extras: torch.Tensor,
    ):
        x = torch.concat((data, extras), dim=-1)
        values = self.critic.forward(x)
        means_stds = self.actor.forward(x)
        dist = self._get_distribution(means_stds)
        return dist, values

    def policy(self, obs: torch.Tensor, extras: torch.Tensor):
        x = torch.concat((obs, extras), dim=-1)
        means_and_stds = self.actor.forward(x)
        return self._get_distribution(means_and_stds)

    @property
    def value_parameters(self):
        return self.critic.parameters()

    @property
    def policy_parameters(self):
        return self.actor.parameters()
