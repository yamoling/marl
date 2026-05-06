import math
from abc import abstractmethod
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING

import torch
from marlenv import DiscreteMARLEnv, Observation

from .nn import NN, RecurrentNN

if TYPE_CHECKING:
    from marl.env import EnvConfig


@dataclass
class QNetwork(NN):
    """
    Takes as input observations of the environment and outputs Q-values for each action.
    """

    n_actions: int
    obs_shape: tuple[int, ...]
    extras_shape: tuple[int, ...]
    _: KW_ONLY
    output_shape: tuple[int, ...] = field(init=False)
    n_objectives: int = 1

    def __post_init__(self):
        if self.n_objectives == 1:
            self.output_shape = (self.n_actions,)
        else:
            self.output_shape = (self.n_actions, self.n_objectives)
        super().__post_init__()
        if self.n_objectives == 1:
            self.action_dim = -1
        else:
            self.action_dim = -2

    @property
    def is_multi_objective(self):
        return len(self.output_shape) > 1

    @property
    def obs_size(self):
        return math.prod(self.obs_shape)

    @property
    def extras_size(self):
        return math.prod(self.extras_shape)

    def qvalues(self, obs: Observation) -> torch.Tensor:
        """
        Compute the Q-values (one per agent, per action and per objective).
        """
        obs_tensor, extra_tensor = obs.as_tensors(self.device)
        qvalues = self.forward(obs_tensor.unsqueeze(0), extra_tensor.unsqueeze(0))
        return qvalues.squeeze(0)

    @abstractmethod
    def forward(self, obs: torch.Tensor, extras: torch.Tensor, /, **kwargs) -> torch.Tensor:
        """
        Compute the Q-values.

        This function should output qvalues of shape (batch_size, n_actions, n_objectives).
        """

    def batch_forward(self, obs: torch.Tensor, extras: torch.Tensor, /, **kwargs) -> torch.Tensor:
        """Compute the Q-values for a batch of observations during training"""
        return self.forward(obs, extras, **kwargs)

    def to_softmax_actor(self):
        from .actor_critic import DiscreteActor

        class ActorFromQNet(DiscreteActor):
            def __init__(self, qnet: QNetwork):
                super().__init__(qnet.output_shape)
                self.qnet = qnet

            def __hash__(self):
                return hash(self.name)

            def logits(self, obs: torch.Tensor, extras: torch.Tensor, available_actions: torch.Tensor | None = None) -> torch.Tensor:
                logits = self.qnet.forward(obs, extras)
                if available_actions is not None:
                    logits = logits.masked_fill(~available_actions, -torch.inf)
                return logits

        return ActorFromQNet(self)

    @classmethod
    def from_env(cls, env: EnvConfig[DiscreteMARLEnv] | DiscreteMARLEnv, **kwargs):
        return cls(env.n_actions, env.observation_shape, env.extras_shape, **kwargs)


@dataclass
class RecurrentQNetwork(QNetwork, RecurrentNN):
    def __post_init__(self):
        QNetwork.__post_init__(self)
        RecurrentNN.__post_init__(self)

    def batch_forward(self, obs: torch.Tensor, extras: torch.Tensor, /, **kwargs) -> torch.Tensor:
        """
        Compute the Q-values for a batch of observations (multiple episodes) during training.

        In this case, the RNN considers hidden states=None.
        """
        saved_hidden_states = self.hidden_states
        self.reset_hidden_states()
        qvalues = self.forward(obs, extras, **kwargs)
        self.hidden_states = saved_hidden_states
        return qvalues
