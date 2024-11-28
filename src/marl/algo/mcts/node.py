import math
from marlenv import State
import numpy as np


class Node:
    parent: "Node | None"
    reward: float
    action: tuple[int, ...]
    """The action taken from the parent to reach the current node."""
    num_visits: int
    total_value: float
    is_terminal: bool
    current_player: int
    children: list["Node"]

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
        # if parent is None:
        #    parent_value = 0.0
        #    sign = 1.0
        # else:
        #    parent_value = parent.reward
        #    sign = -1.0 if parent.current_player == 0 else 1.0
        # self.reward = sign * parent_value + reward
        if parent is None:
            self.reward = 0
        else:
            self.reward = reward if parent.current_player == 0 else -reward
        self.action = action_taken or tuple()
        self.num_visits = 0
        self.total_value = 0.0
        self.is_terminal = is_terminal
        self.current_player = turn
        self.children = []

    @staticmethod
    def root(state: State[np.ndarray]):
        root = Node(state, None, None, False, 0, 0.0)
        # The number of visits is initializes to 1 to be expanded on the first iteration
        # root.num_visits = 1
        return root

    @property
    def avg_value(self):
        if self.num_visits == 0:
            return 0.0
        return self.total_value / self.num_visits

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def is_root(self):
        return self.parent is None

    @property
    def best_action(self):
        return max(self.children, key=lambda item: item.avg_value).action

    def backpropate(self, child_value: float, gamma: float):
        self.num_visits += 1
        if self.parent is None:
            # It is not necessary to compute anything for the root node
            return

        # Since we work with MDPs where the reward happens on transitions,
        # the value that has to be notified to the parent is the sum of the
        # discounted child value with the reward that was obtained on the
        # transition.
        future_returns = self.reward + gamma * child_value
        self.total_value += future_returns
        self.parent.backpropate(future_returns, gamma)

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
