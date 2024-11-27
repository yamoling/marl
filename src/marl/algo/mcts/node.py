import math
from marlenv import State
import numpy as np


class Node:
    def __init__(
        self,
        state: State[np.ndarray],
        action_taken: tuple[int, ...] | None,
        parent: "Node | None",
        is_terminal: bool,
        turn: int,
        reward: float,
    ):
        self.state = state
        self.parent = parent
        if parent is None:
            parent_value = 0.0
            sign = 1.0
        else:
            parent_value = parent.value
            sign = -1.0 if parent.current_player == 0 else 1.0
        self.value = sign * parent_value + reward
        self.action = action_taken or tuple()
        """The action taken from the parent to reach the current node."""
        self.num_visits = 0
        self.total_value = 0.0
        self.is_terminal = is_terminal
        self.current_player = turn
        self.children = list[Node]()
        """The reward that the player received when expanding the node."""

    @staticmethod
    def root(state: State[np.ndarray]):
        return Node(state, None, None, False, 0, 0.0)

    @property
    def avg_value(self):
        if self.num_visits == 0:
            return 0.0
        return self.total_value / self.num_visits

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def best_action(self):
        return max(self.children, key=lambda item: item.avg_value).action

    def backpropate(self, value: float, gamma: float):
        self.total_value += value
        self.num_visits += 1
        if self.parent is not None:
            self.parent.backpropate(gamma * value, gamma)

    def ucb1(self, exploration_value: float) -> float:
        assert self.parent is not None, "Can not calculate UCB1 for root node"
        if self.num_visits == 0:
            return float("inf")
        return self.avg_value + math.sqrt(exploration_value * math.log(self.parent.num_visits) / self.num_visits)

    def get_child_with_state(self, state: State[np.ndarray], max_depth: int) -> "Node | None":
        if max_depth == 0:
            return None
        for child in self.children:
            if child.state == state:
                return child
            found = child.get_child_with_state(state, max_depth - 1)
            if found is not None:
                return found
        return None
