from dataclasses import dataclass
from marlenv import Observation
from abc import abstractmethod
import torch

from marlenv.models import MARLEnv, DiscreteActionSpace

from .nn import NN, RecurrentNN


@dataclass
class QNetwork(NN):
    """
    Takes as input observations of the environment and outputs Q-values for each action.
    """

    def __init__(self, input_shape: tuple[int, ...], extras_shape: tuple[int, ...], output_shape: tuple[int, ...]):
        super().__init__(input_shape, extras_shape, output_shape)
        match output_shape:
            case (_,):
                self.action_dim = -1
                self.is_multi_objective = False
                """The action dimention when predicting qvalues. The value is -1 for single objective RL and -2 for multi-objective RL."""
            case (_, _):
                self.action_dim = -2
                self.is_multi_objective = True
            case other:
                raise ValueError(f"Cannot compute action_dim for output_shape: {other}")

    def to_tensor(self, obs: Observation) -> tuple[torch.Tensor, torch.Tensor]:
        extras = torch.from_numpy(obs.extras).unsqueeze(0).to(self.device)
        obs_tensor = torch.from_numpy(obs.data).unsqueeze(0).to(self.device)
        return obs_tensor, extras

    def qvalues(self, obs: Observation) -> torch.Tensor:
        """
        Compute the Q-values (one per agent, per action and per objective).
        """
        obs_tensor, extra_tensor = self.to_tensor(obs)
        qvalues = self.forward(obs_tensor, extra_tensor)
        return qvalues.squeeze(0)

    def value(self, obs: Observation) -> torch.Tensor:
        """Compute the value function (maximal q-value of each agent)."""
        objective_qvalues = self.qvalues(obs)
        qvalues = torch.sum(objective_qvalues, dim=-1)
        agent_values = qvalues.max(dim=-1).values
        return agent_values.mean(dim=-1)

    @abstractmethod
    def forward(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        """
        Compute the Q-values.

        This function should output qvalues of shape (batch_size, n_actions, n_objectives).
        """

    def batch_forward(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        """Compute the Q-values for a batch of observations during training"""
        return self.forward(obs, extras)

    def set_testing(self, test_mode: bool = True):
        """Set the network in testing mode"""
        self.test_mode = test_mode

    @classmethod
    def from_env[A](cls, env: MARLEnv[A, DiscreteActionSpace]):
        if env.reward_space.size == 1:
            output_shape = (env.n_actions,)
        else:
            output_shape = (env.n_actions, env.reward_space.size)
        return cls(
            input_shape=env.observation_shape,
            extras_shape=env.extras_shape,
            output_shape=output_shape,
        )


@dataclass
class RecurrentQNetwork(QNetwork, RecurrentNN):
    def __init__(self, input_shape: tuple[int, ...], extras_shape: tuple[int, ...], output_shape: tuple[int, ...]):
        QNetwork.__init__(self, input_shape, extras_shape, output_shape)
        RecurrentNN.__init__(self, input_shape, extras_shape, output_shape)

    def value(self, obs: Observation) -> torch.Tensor:
        """Compute the value function. Does not update the hidden states."""
        hidden_states = self.hidden_states
        objective_qvalues = self.qvalues(obs)
        qvalues = torch.sum(objective_qvalues, dim=-1)
        agent_values = torch.max(qvalues, dim=-1).values
        self.hidden_states = hidden_states
        return agent_values.mean(dim=-1)

    def batch_forward(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        """
        Compute the Q-values for a batch of observations (multiple episodes) during training.

        In this case, the RNN considers hidden states=None.
        """
        self.test_mode = False
        saved_hidden_states = self.hidden_states
        self.reset_hidden_states()
        qvalues = self.forward(obs, extras)
        self.hidden_states = saved_hidden_states
        return qvalues
