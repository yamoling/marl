import random
from dataclasses import dataclass
from typing import Sequence

import torch
from marlenv import MARLEnv
from marlenv.utils import Schedule
from torch import Tensor
from torch.distributions import Categorical
from torch.nn import ModuleList

from marl.models.nn import NN, Actor, QNetwork
from marl.models.nn.options import OptionCriticNetwork
from marl.nn.model_bank.generic import CNN


@dataclass(unsafe_hash=True)
class CNNOptionCritic(OptionCriticNetwork):
    policies: torch.nn.ModuleList
    q_options: QNetwork
    options_termination: NN

    def __init__(self, policies: torch.nn.ModuleList, q_options: QNetwork, options_termination: NN):
        assert len(q_options.output_shape) == 1, "Multi-objective options are not supported"
        OptionCriticNetwork.__init__(self, q_options.output_shape[0])
        self.policies = policies
        self.q_options = q_options
        self.options_termination = options_termination

    @staticmethod
    def from_env(env: MARLEnv, n_options: int):
        from marl.nn.model_bank.actor_critics import CNNActor
        from marl.nn.model_bank.qnetworks import QCNN

        assert len(env.observation_shape) == 3
        policies = torch.nn.ModuleList([CNNActor(env.observation_shape, env.extras_size, env.n_actions) for _ in range(n_options)])
        terminations = CNN(env.observation_shape, env.extras_size, n_options, output_activation="sigmoid")
        q_options = QCNN(env.observation_shape, env.extras_size, n_options)
        return CNNOptionCritic(policies, q_options, terminations)

    def compute_q_options(self, obs: Tensor, extras: Tensor) -> Tensor:
        return self.q_options.forward(obs, extras)

    def termination_probability(self, obs: Tensor, extras: Tensor, options: Tensor) -> Tensor:
        probs = self.options_termination.forward(obs, extras)
        while options.ndim < probs.ndim:
            options = options.unsqueeze(-1)
        probs = torch.gather(probs, dim=-1, index=options)
        # Squeeze the last dimension introduced by the gathering
        return probs.squeeze(-1)

    def policy(
        self,
        obs: Tensor,
        extras: Tensor,
        available_actions: torch.Tensor,
        options: Sequence[int] | torch.Tensor,
    ):
        if not isinstance(options, Tensor):
            logits = [self.policies[option].forward(obs, extra) for option, obs, extra in zip(options, obs, extras)]
            logits = torch.stack(logits)
        else:
            # To avoid looping on the whole batch, we perform one forward pass for each policy and
            # then gather the relevant logits according to the options tensor.
            logits = torch.stack([policy.forward(obs, extras) for policy in self.policies])
            # Turn the options of shape (batch_size, n_agents, n_actions) to (1, batch_size, n_agents, n_actions) to gather the correct logits for each agent and option
            index = options.unsqueeze(0).expand(1, -1, -1, available_actions.shape[-1])
            logits = torch.gather(logits, dim=0, index=index).squeeze(0)
        logits[~available_actions] = -torch.inf
        dist = torch.distributions.Categorical(logits=logits)
        return dist


@dataclass
class SimpleOptionCritic(Actor):
    """
    Vanilla Option-Critic adapted for multi-agent. In this Option-Critic implementation, each agent has its own option.

    Article by Bacon, Harb & Precup: https://arxiv.org/pdf/1609.05140
    """

    policies: ModuleList
    q_options: QNetwork
    options_termination: NN
    n_options: int
    n_agents: int
    epsilon: Schedule
    temperature: float
    options: list[int]

    def __init__(
        self,
        policies: list[Actor],
        q_options: QNetwork,
        options_termination: NN,
        n_agents: int,
        epsilon: Schedule = Schedule.constant(0.1),
        temperature: float = 1.0,
    ):
        super().__init__()
        self.policies = ModuleList(policies)
        self.q_options = q_options
        self.options_termination = options_termination
        self.n_options = len(policies)
        self.n_agents = n_agents
        self.epsilon = epsilon
        self.temperature = temperature
        self.options = [random.randint(0, self.n_options - 1) for _ in range(self.n_agents)]

    def value_upon_arrival(self, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        """
        Equation (3) of the paper:
            U(omega, s) = (1 - beta(omega, s)) * Q(omega, s) + beta(omega, s) * V(omega', s)
        """
        probs = self.compute_termination_probs(obs, extras)
        q_options = self.compute_q_options(obs, extras)
        values = []
        for agent_option, terminations, q_option in zip(self.options, probs, q_options):
            current_q_option = q_option[agent_option].item()
            best_q_option = q_option.max().item()
            term_prob = terminations[agent_option].item()
            values.append((1 - term_prob) * current_q_option + term_prob * best_q_option)
        return torch.tensor(values)

    def compute_q_options(self, obs: Tensor, extras: Tensor):
        return self.q_options.forward(obs, extras)

    def compute_termination_probs(self, obs: Tensor, extras: Tensor) -> Tensor:
        return self.options_termination.forward(obs, extras)

    def policy(self, obs: torch.Tensor, extras: Tensor, available_actions: Tensor):
        logits = []
        obs, extras, available_actions = obs.squeeze(0), extras.squeeze(0), available_actions.squeeze(0)
        for agent_num, option in enumerate(self.options):
            logits.append(self.policies[option].forward(obs[agent_num], extras[agent_num]))
        logits = torch.stack(logits)
        logits = self.mask(logits, available_actions)
        action_dist = (logits / self.temperature).softmax(dim=-1)
        return Categorical(action_dist)

    def update_current_option(self, obs: Tensor, extras: Tensor):
        termination_probs = self.options_termination.forward(obs, extras)
        greedy_options = self._compute_greedy_option(obs, extras)
        for agent_num, agent_option in enumerate(self.options):
            is_terminated = random.random() < termination_probs[agent_num, agent_option].item()
            if is_terminated:
                if random.random() < self.epsilon:
                    self.options[agent_num] = random.randint(0, self.n_options - 1)
                else:
                    self.options[agent_num] = greedy_options[agent_num]

    def _compute_greedy_option(self, obs, extras) -> list[int]:
        q_policies = self.compute_q_options(obs, extras)
        return q_policies.argmax(dim=-1).tolist()

    def forward(self, obs: Tensor, extras: Tensor, *args, **kwargs):
        return self.policy(obs, extras, *args, **kwargs)

    def __hash__(self):
        return hash(self.name)


class OptionTermination(NN):
    def __init__(self, n_options: int, obs_shape: tuple[int, int, int], extras_shape: tuple[int, ...]):
        super().__init__()
        assert len(extras_shape) == 1
        self.cnn = CNN(obs_shape, extras_shape[0], n_options)

    def forward(self, obs: Tensor, extras: Tensor, *args, **kwargs):
        output = self.cnn.forward(obs, extras)
        return torch.nn.functional.sigmoid(output)
