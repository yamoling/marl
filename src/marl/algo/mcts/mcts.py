import math
from copy import deepcopy
import time
from marlenv import MARLEnv, DiscreteActionSpace, State
import numpy as np
import random


class TreeNode:
    def __init__(
        self,
        state: State[np.ndarray],
        action_taken: tuple[int, ...] | None,
        parent: "TreeNode | None",
        is_terminal: bool,
        turn: int,
        reward: float,
        gamma: float,
    ):
        self.state = state
        self.parent = parent
        if parent is None:
            self.value = 0
        else:
            if parent.current_player == 0:
                self.value = parent.value + reward * gamma
            else:
                self.value = parent.value - reward * gamma
        self.action = action_taken or tuple()
        """The action taken from the parent to reach the current node."""
        self.num_visits = 0
        self.total_value = 0.0
        self.is_terminal = is_terminal
        self.current_player = turn
        self.children = list[TreeNode]()
        """The reward that the player received when expanding the node."""

    @staticmethod
    def root(state: State[np.ndarray]):
        return TreeNode(state, None, None, False, 0, 0.0, 0.0)

    @staticmethod
    def node(state: State[np.ndarray], parent: "TreeNode", action_taken: tuple[int, ...], turn: int, reward: float, gamma: float):
        return TreeNode(state, action_taken, parent, False, turn, reward, gamma)

    @staticmethod
    def leaf(state: State[np.ndarray], parent: "TreeNode", action_taken: tuple[int, ...], turn: int, reward: float, gamma: float):
        return TreeNode(state, action_taken, parent, True, turn, reward, gamma)

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

    def random_simulation(self, env: MARLEnv[DiscreteActionSpace], gamma: float, n_adversaries: int) -> float:
        if self.is_terminal:
            # Note: the assumes that the transition function is deterministic
            return self.value

        env.set_state(self.state)
        done = False
        total_reward = 0.0
        current_gamma = 1.0
        turn = self.current_player
        n_adversarial_players = n_adversaries + 1
        while not done:
            action = env.action_space.sample(env.available_actions())
            step = env.step(action)
            done = step.done
            # We assume that the maximizing player is always the first one
            # because the search initializes the root node with the maximizing player (0).
            if turn == 0:
                total_reward += step.reward * current_gamma
            else:
                total_reward -= step.reward * current_gamma
            current_gamma *= gamma
            turn = (turn + 1) % n_adversarial_players
        return total_reward

    def select(self, exploration_constant: float) -> "TreeNode":
        """Select the node to expand."""
        if self.is_leaf:
            return self
        # child = self.ucb1_policy(exploration_constant)
        child = random.choice(self.children)
        return child.select(exploration_constant)

    def train(self, env: MARLEnv[DiscreteActionSpace], exploration_constant: float, gamma: float, n_adversaries: int):
        node = self.select(exploration_constant)
        # Intuition: if the node has never been sampled, we do the rollout from there without expanding
        if node.num_visits > 0:
            node = node.expand(env, n_adversaries, gamma)
        value = node.random_simulation(env, gamma, n_adversaries)
        node.backpropate(value, gamma)

    def ucb1_policy(self, exploration_value: float):
        best_child = self.children[0]
        best_value = best_child.ucb1(exploration_value)
        for child in self.children[1:]:
            node_value = child.ucb1(exploration_value)
            if node_value > best_value:
                best_value = node_value
                best_child = child
        return best_child

    def expand(self, env: MARLEnv[DiscreteActionSpace], n_adversaries: int, gamma: float) -> "TreeNode":
        env.set_state(self.state)
        next_player = (self.current_player + 1) % (n_adversaries + 1)
        for action in env.available_joint_actions():
            env.set_state(self.state)
            step = env.step(action)
            child = TreeNode(
                step.state,
                action,
                self,
                step.is_terminal,
                next_player,
                step.reward,
                gamma,
            )
            self.children.append(child)
        if len(self.children) == 0:
            return self
        return self.children[0]


def search(
    env: MARLEnv[DiscreteActionSpace],
    time_limit_ms: int | None = None,
    iteration_limit: int | None = None,
    exploration_constant: float = 2**0.5,
    gamma: float = 0.99,
    n_adversaries: int = 1,
):
    """
    This implementaiton assumes that:
     - transitions are deterministic
     - rewards are deterministic
    """
    env = deepcopy(env)
    root = TreeNode.root(env.get_state())
    root.expand(env, n_adversaries, gamma)
    match (time_limit_ms, iteration_limit):
        case (None, None):
            raise ValueError("Must have either a time limit or an iteration limit")
        case (time_limit_ms, None):
            deadline = time.time() + time_limit_ms / 1000
            while time.time() < deadline:
                root.train(env, exploration_constant, gamma, n_adversaries)
        case (None, iteration_limit):
            for _ in range(iteration_limit):
                root.train(env, exploration_constant, gamma, n_adversaries)
        case _:
            raise ValueError("Cannot have both a time limit and an iteration limit")
    for child in root.children:
        print(f"Action: {int(child.action[0])}, Value: {child.avg_value:.5f}")
    return root.best_action
