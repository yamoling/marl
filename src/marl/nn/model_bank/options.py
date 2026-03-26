import random
from dataclasses import dataclass

import torch
from marlenv.utils import Schedule
from torch import Tensor
from torch.distributions import Categorical
from torch.nn import ModuleList

from marl.models.nn import NN, Actor, QNetwork
from marl.nn.model_bank.generic import CNN


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
