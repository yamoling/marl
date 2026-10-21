import random
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, Literal, cast

import numpy as np
import torch
from marlenv import MARLEnv
from torch import Tensor
from torch.distributions import Categorical
from torch.nn import ModuleList

from marl.models.nn import NN, Actor, CategoricalActor, QNetwork
from marl.models.nn.options import OptionCriticNetwork
from marl.nn.model_bank.generic import CNN, MLP
from marl.utils import Schedule


@dataclass
class CNNOptionCritic(OptionCriticNetwork):
    policies: torch.nn.ModuleList
    q_options: QNetwork
    options_termination: NN

    def __init__(self, policies: torch.nn.ModuleList, q_options: QNetwork, options_termination: NN):
        assert len(q_options.output_shape) == 1, "Multi-objective options are not supported"
        super().__init__((1,), q_options.n_actions)
        self.policies = policies
        self.q_options = q_options
        self.options_termination = options_termination

    def to_dict(self):
        """Serialize registered heads as recipes, not raw PyTorch modules. @ai-generated"""
        return {
            "class-name": self.__class__.__name__,
            "name": self.name,
            "policies": [cast(CategoricalActor, policy).to_dict() for policy in self.policies],
            "q_options": self.q_options.to_dict(),
            "options_termination": self.options_termination.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], *, exact_type: bool = False):
        """Reconstruct the module list and registered value/termination heads. @ai-generated"""
        policies = ModuleList([CategoricalActor.from_dict(dict(policy)) for policy in d["policies"]])
        return cls(policies, QNetwork.from_dict(dict(d["q_options"])), NN.from_dict(dict(d["options_termination"])))

    def materialize(self, root, *, _seen: set[int] | None = None):
        """Visit registered serializable heads without indexing Module internals. @ai-generated"""
        if _seen is None:
            _seen = set()
        if id(self) in _seen:
            return
        _seen.add(id(self))
        for child in (*self.policies, self.q_options, self.options_termination):
            cast(NN, child).materialize(root, _seen=_seen)

    @classmethod
    def from_env(cls, env: MARLEnv, n_options: int, **kwargs):
        """Build compatible convolutional policy, value, and termination heads. @ai-edited"""
        from marl.nn.model_bank.actor_critics import CategoricalConvActor
        from marl.nn.model_bank.qnetworks import QCNN

        assert len(env.observation_shape) == 3
        policies = torch.nn.ModuleList([CategoricalConvActor.from_env(env, **kwargs) for _ in range(n_options)])
        terminations = OptionTermination((n_options,), n_options, env.observation_shape, env.extras_shape)
        q_options = QCNN(n_options, env.n_agents, env.observation_shape, env.extras_shape, duelling=False)
        return cls(policies, q_options, terminations)

    def compute_q_options(self, obs: Tensor, extras: Tensor) -> Tensor:
        return self.q_options.batch_qvalues(obs, extras)

    def termination_probability(self, obs: Tensor, extras: Tensor, options: Tensor) -> Tensor:
        probs = self.options_termination.forward(obs, extras)
        while options.ndim < probs.ndim:
            options = options.unsqueeze(-1)
        probs = torch.gather(probs, dim=-1, index=options)
        # Squeeze the last dimension introduced by the gathering
        return probs.squeeze(-1)

    def policy(self, obs: Tensor, extras: Tensor, available_actions: torch.Tensor, options: Sequence[int] | torch.Tensor):
        """
        Select agent-wise option logits without dropping leading batch axes.

        Only the policies of the options that are actually active are evaluated (and back-propagated through);
        the logits of the other options would be discarded anyway.

        @ai-edited
        """
        if isinstance(options, torch.Tensor):
            active = torch.unique(options).tolist()
        else:
            active = np.unique(np.asarray(options)).tolist()
        options = torch.as_tensor(options, dtype=torch.long).to(obs.device, non_blocking=True)
        if options.ndim == available_actions.ndim and options.shape[-1] == 1:
            options = options.squeeze(-1)
        options = torch.broadcast_to(options, available_actions.shape[:-1])
        logits = self.policies[active[0]].forward(obs, extras)
        for option in active[1:]:
            logits = torch.where((options == option).unsqueeze(-1), self.policies[option].forward(obs, extras), logits)
        return Categorical(logits=logits.masked_fill(~available_actions, -torch.inf))

    def __hash__(self):
        return id(self)


@dataclass
class SimpleOptionCritic(Actor[torch.distributions.Categorical]):
    """
    Vanilla Option-Critic adapted for multi-agent. In this Option-Critic implementation, each agent has its own option.

    Article by Bacon, Harb & Precup: https://arxiv.org/pdf/1609.05140
    """

    n_options: int
    options_policy: CategoricalActor
    """The policy of one single option that is replicated (with different weights) for all options."""
    q_options: QNetwork
    options_termination: NN
    _: KW_ONLY
    epsilon: Schedule = field(default_factory=lambda: Schedule.constant(0.1))
    temperature: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        self.policies = ModuleList([self.options_policy] + [deepcopy(self.options_policy) for p in range(self.n_options - 1)])
        self.current_options = [random.randint(0, self.n_options - 1) for _ in range(self.n_agents)]
        """The options currently selected by the agents."""

    def value_upon_arrival(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        """
        Equation (3) of the paper:
            U(omega, s) = (1 - beta(omega, s)) * Q(omega, s) + beta(omega, s) * V(omega', s)
        """
        probs = self.compute_termination_probs(obs, extras)
        q_options = self.compute_q_options(obs, extras)
        values = []
        for agent_option, terminations, q_option in zip(self.current_options, probs, q_options):
            current_q_option = q_option[agent_option].item()
            best_q_option = q_option.max().item()
            term_prob = terminations[agent_option].item()
            values.append((1 - term_prob) * current_q_option + term_prob * best_q_option)
        return torch.tensor(values)

    def compute_q_options(self, obs: Tensor, extras: Tensor):
        return self.q_options.forward(obs, extras)

    def compute_termination_probs(self, obs: Tensor, extras: Tensor) -> Tensor:
        return self.options_termination.forward(obs, extras)

    def policy(self, obs: torch.Tensor, extras: Tensor, *, available_actions: Tensor | None = None, **kwargs):
        assert available_actions is not None, "Available actions must be provided to compute the policy"
        logits = []
        obs, extras, available_actions = obs.squeeze(0), extras.squeeze(0), available_actions.squeeze(0)
        for agent_num, option in enumerate(self.current_options):
            logits.append(self.policies[option].forward(obs[agent_num], extras[agent_num]))
        logits = torch.stack(logits)
        logits = self.mask(logits, available_actions)
        action_dist = (logits / self.temperature).softmax(dim=-1)
        return Categorical(action_dist)

    def update_current_option(self, obs: Tensor, extras: Tensor):
        termination_probs = self.options_termination.forward(obs, extras)
        greedy_options = self._compute_greedy_option(obs, extras)
        for agent_num, agent_option in enumerate(self.current_options):
            is_terminated = random.random() < termination_probs[agent_num, agent_option].item()
            if is_terminated:
                if random.random() < self.epsilon:
                    self.current_options[agent_num] = random.randint(0, self.n_options - 1)
                else:
                    self.current_options[agent_num] = greedy_options[agent_num]

    def _compute_greedy_option(self, obs, extras) -> list[int]:
        q_policies = self.compute_q_options(obs, extras)
        return q_policies.argmax(dim=-1).tolist()

    def forward(self, obs: Tensor, extras: Tensor, *args, **kwargs):
        return self.policy(obs, extras, *args, **kwargs)

    def randomize(self, method: Literal["xavier", "orthogonal"] | Callable[[torch.Tensor], Any] = "xavier"):
        self.current_options = [random.randint(0, self.n_options - 1) for _ in range(self.n_agents)]
        return super().randomize(method)

    def __hash__(self):
        return id(self)


@dataclass
class OptionTermination(NN):
    n_options: int
    obs_shape: tuple[int, int, int]
    extras_shape: tuple[int, ...]

    def __post_init__(self):
        super().__post_init__()
        self.cnn = CNN(self.obs_shape)
        self.mlp = MLP((self.n_options,), self.cnn.output_size, self.extras_shape[0], output_activation="sigmoid")

    def forward(self, obs: Tensor, extras: Tensor, *args, **kwargs):
        return self.mlp(self.cnn(obs), extras)

    def __hash__(self):
        return id(self)
