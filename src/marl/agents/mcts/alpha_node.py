from marlenv import State, MARLEnv, DiscreteActionSpace
import torch
import numpy as np
import math
import random
from marl.models.nn import DiscreteActorCriticNN


class AlphaNode:
    prior: float
    value: float
    state: State
    parent: "AlphaNode | None"
    children: list["AlphaNode"]
    action: int
    is_terminal: bool
    visit_count: int
    value_sum: float

    def __init__(
        self,
        state: State,
        parent: "AlphaNode | None",
        action: int,
        is_terminal: bool,
        prior: float,
        value: float,
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.is_terminal = is_terminal
        self.value = value
        self.children = []
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0

    @staticmethod
    def root(state: State, value: float):
        return AlphaNode(state, None, -1, False, 0.0, value)

    def get_max_ucb_child(self, c: float):
        """
        Returns the child with the highest UCB value.
        Used for selecting the next node to explore.
        """
        if self.is_terminal:
            raise ValueError("There are no children for a terminal node")
        if len(self.children) == 0:
            raise ValueError("This node is not yet expanded !")
        max_child = self.children[0]
        max_ucb = max_child.ucb(c)
        for child in self.children[1:]:
            ucb = child.ucb(c)
            if ucb > max_ucb:
                max_ucb = ucb
                max_child = child
        return max_child

    @property
    def q_value(self):
        """Estimated Q-value"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def target_prob(self, temprature: float):
        """
        Target probability of taking this action according to the MCTS policy.
        This should be used to train the neural network.
        """
        assert self.parent is not None, "It does not make sense to calculate the probability of taking the root node."
        exponent = 1 / temprature
        return self.visit_count**exponent / self.parent.visit_count**exponent

    @property
    def best_child(self):
        """Best child when exploiting the tree."""
        return max(self.children, key=lambda child: child.visit_count)

    def children_probs(self, tau: float) -> np.ndarray:
        """
        Return the probabilities of selecting each child according to the MCTS policy.
        """
        exponent = 1 / tau
        visit_counts = np.array([child.visit_count for child in self.children])
        numerator = visit_counts**exponent
        denominator = self.visit_count**exponent
        return numerator / denominator

    def get_child(self, temperature: float):
        """Return the child to select according to the MCTS policy."""
        if self.is_terminal:
            raise ValueError("There are no children for a terminal node")
        if len(self.children) == 0:
            raise ValueError("This node is not yet expanded !")
        if temperature == 0.0:
            return self.best_child
        probs = list(self.children_probs(temperature))
        return random.choices(self.children, weights=probs, k=1)[0]

    def ucb(self, c: float):
        """Upper Confidence Bound in the context of AlphaZero."""
        assert self.parent is not None
        return self.q_value + c * self.prior * math.sqrt(self.parent.visit_count) / (self.visit_count + 1)

    def expand(self, env: MARLEnv[list[int], DiscreteActionSpace], nn: DiscreteActorCriticNN, gamma: float):
        if self.is_terminal:
            return
        env.set_state(self.state)
        state = torch.from_numpy(self.state.data).unsqueeze(0).to(nn._device)
        extras = torch.from_numpy(self.state.extras).unsqueeze(0).to(nn._device)
        available = torch.from_numpy(env.available_actions()).to(nn._device)
        with torch.no_grad():
            priors = nn.policy(state, extras, available)[0].tolist()
        for action, prior in enumerate(priors):
            if prior == 0.0:
                continue
            env.set_state(self.state)
            step = env.step([action])
            if step.is_terminal:
                next_value = 0.0
            else:
                next_state = torch.from_numpy(step.state.data).unsqueeze(0).to(nn._device)
                next_extras = torch.from_numpy(step.state.extras).unsqueeze(0).to(nn._device)
                with torch.no_grad():
                    next_value = nn.value(next_state, next_extras).item()
            self.children.append(
                AlphaNode(
                    step.state,
                    self,
                    action,
                    step.is_terminal,
                    prior,
                    step.reward.item() + gamma * next_value,
                )
            )

    def backprop(self, value: float):
        self.visit_count += 1
        self.value_sum += value
        if self.parent is not None:
            self.parent.backprop(value)

    @property
    def is_expanded(self):
        if self.is_terminal:
            return False
        return len(self.children) > 0
