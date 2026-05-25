import math
from abc import abstractmethod
from dataclasses import KW_ONLY, dataclass, field

import torch
from marlenv import DiscreteMARLEnv, Observation

from marl.env import EnvConfig

from .nn import NN, RecurrentNN


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
    noisy: bool = False
    duelling: bool = True
    """
    When enabled, the network outputs a 1 value and `self.n_actions` advantages.

    Paper: https://proceedings.mlr.press/v48/wangf16.pdf
    """
    independent: bool = False
    """Whether each agent has its own independent weights or not."""

    def __post_init__(self):
        n_action_outputs = self.n_actions
        if self.duelling:
            n_action_outputs += 1
        if self.n_objectives == 1:
            self.output_shape = (n_action_outputs,)
        else:
            self.output_shape = (n_action_outputs, self.n_objectives)
        # ⚠️ self.output_shape must be set BEFORE calling super().__post_init__() !
        super().__post_init__()
        if self.n_objectives == 1:
            self.action_dim = -1
        else:
            self.action_dim = -2
        if self.duelling and self.n_objectives != 1:
            raise NotImplementedError("Multi-objective is currently not supported with duelling DQN.")

    def _get_qvalues(self, outputs: torch.Tensor):
        if not self.duelling:
            return outputs
        if outputs.ndim == 3:
            value = outputs[:, :, -1].unsqueeze(-1)  # Unsqueeze to keep 3 dimensions (batch_size, n_agents, 1)
            adv = outputs[:, :, :-1]
        elif outputs.ndim == 4:
            value = outputs[:, :, :, -1].unsqueeze(-1)
            adv = outputs[:, :, :, :-1]
        else:
            raise NotImplementedError()
        mean_adv = torch.mean(adv, dim=-1, keepdim=True)
        res = value + adv - mean_adv
        return res

    @property
    def is_multi_objective(self):
        return self.n_objectives > 1

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
        outputs = self.forward(obs_tensor.unsqueeze(0), extra_tensor.unsqueeze(0))
        qvalues = self._get_qvalues(outputs)
        return qvalues.squeeze(0)

    @abstractmethod
    def forward(self, obs: torch.Tensor, extras: torch.Tensor, /, **kwargs) -> torch.Tensor:
        """
        Forward the observation and extras through the network. The output is either
        directly the Q-values if self.duelling is False or the value and advantages otherwise.
        """

    def batch_qvalues(self, obs: torch.Tensor, extras: torch.Tensor, /, **kwargs) -> torch.Tensor:
        """Compute the Q-values for a batch of observations, typically used in batched training loops."""
        outputs = self.forward(obs, extras, **kwargs)
        return self._get_qvalues(outputs)

    def to_softmax_actor(self):
        from .actor_critic import DiscreteActor

        class ActorFromQNet(DiscreteActor):
            def __init__(self, qnet: QNetwork):
                super().__init__(qnet.output_shape)
                self.qnet = qnet

            def __hash__(self):
                return hash(self.name)

            def logits(self, obs: torch.Tensor, extras: torch.Tensor, available_actions: torch.Tensor | None = None) -> torch.Tensor:
                logits = self.qnet.batch_qvalues(obs, extras)
                if available_actions is not None:
                    logits = logits.masked_fill(~available_actions, -torch.inf)
                return logits

        return ActorFromQNet(self)

    @classmethod
    def from_env(cls, env: EnvConfig[DiscreteMARLEnv] | DiscreteMARLEnv, *, noisy: bool = False, duelling: bool = False, **kwargs):
        return cls(env.n_actions, env.observation_shape, env.extras_shape, noisy=noisy, duelling=duelling, **kwargs)


@dataclass
class RecurrentQNetwork(QNetwork, RecurrentNN):
    def __post_init__(self):
        QNetwork.__post_init__(self)
        RecurrentNN.__post_init__(self)

    def batch_qvalues(self, obs: torch.Tensor, extras: torch.Tensor, *, masks: torch.Tensor | None, **kwargs) -> torch.Tensor:
        """
        Compute the Q-values for a batch of observations (multiple episodes) during training.

        In this case, the RNN considers hidden states=None.
        """
        saved_hidden_states = self._hidden_states
        self.reset_hidden_states()
        qvalues = super().batch_qvalues(obs, extras, masks=masks, **kwargs)
        self._hidden_states = saved_hidden_states
        return qvalues
