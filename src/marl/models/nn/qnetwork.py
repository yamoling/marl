from abc import abstractmethod
from dataclasses import dataclass

import torch
from marlenv import Observation
from marlenv.models import MARLEnv, MultiDiscreteSpace

from .nn import NN, RecurrentNN


@dataclass
class QNetwork(NN):
    """
    Takes as input observations of the environment and outputs Q-values for each action.
    """

    output_shape: int | tuple[int] | tuple[int, int]

    def __post_init__(self):
        super().__post_init__()
        match self.output_shape:
            case int(n_actions) | (n_actions,):
                self.action_dim = -1
                self.is_multi_objective = False
                self.output_shape = (n_actions,)
            case (_, _):
                self.action_dim = -2
                self.is_multi_objective = True
                self.output_shape = self.output_shape
            case other:
                raise ValueError(f"Cannot compute action_dim for output_shape: {other}")

    def qvalues(self, obs: Observation) -> torch.Tensor:
        """
        Compute the Q-values (one per agent, per action and per objective).
        """
        obs_tensor, extra_tensor = obs.as_tensors(self.device)
        qvalues = self.forward(obs_tensor.unsqueeze(0), extra_tensor.unsqueeze(0))
        return qvalues.squeeze(0)

    def value(self, obs: Observation) -> torch.Tensor:
        """Compute the value function (maximal q-value of each agent)."""
        objective_qvalues = self.qvalues(obs)
        qvalues = torch.sum(objective_qvalues, dim=-1)
        agent_values = qvalues.max(dim=-1).values
        return agent_values.mean(dim=-1)

    @abstractmethod
    def forward(self, obs: torch.Tensor, extras: torch.Tensor, /, **kwargs) -> torch.Tensor:
        """
        Compute the Q-values.

        This function should output qvalues of shape (batch_size, n_actions, n_objectives).
        """

    def batch_forward(self, obs: torch.Tensor, extras: torch.Tensor, /, **kwargs) -> torch.Tensor:
        """Compute the Q-values for a batch of observations during training"""
        return self.forward(obs, extras, **kwargs)

    @classmethod
    def from_env(cls, env: MARLEnv[MultiDiscreteSpace]):
        if env.reward_space.size == 1:
            output_shape = env.n_actions
        else:
            output_shape = (env.n_actions, env.reward_space.size)
        return cls(output_shape=output_shape)


@dataclass
class RecurrentQNetwork(QNetwork, RecurrentNN):
    def __init__(self, output_shape: int | tuple[int, int]):
        QNetwork.__init__(self, output_shape)
        RecurrentNN.__init__(self)

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
        saved_hidden_states = self.hidden_states
        self.reset_hidden_states()
        qvalues = self.forward(obs, extras)
        self.hidden_states = saved_hidden_states
        return qvalues
