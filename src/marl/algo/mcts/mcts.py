from copy import deepcopy
import time
from typing import Literal, Callable
from marlenv import MARLEnv, DiscreteActionSpace, State
import random

from .node import Node


class MTCS:
    cache: Node | None
    """Previous tree used to speed up the search."""
    exploration_constant: float
    gamma: float
    """Discount factor."""
    time_limit_ms: int | None
    iteration_limit: int | None
    n_adversaries: int
    policy: Callable[[Node], Node]
    """Policy to select the next node to explore."""

    def __init__(
        self,
        env: MARLEnv[DiscreteActionSpace],
        exploration_constant: float = 2**0.5,
        gamma: float = 0.99,
        time_limit_ms: int | None = None,
        iteration_limit: int | None = None,
        n_adversaries: int = 0,
        policy: Literal["ucb", "random"] = "ucb",
    ):
        self.env = deepcopy(env)
        self.exploration_constant = exploration_constant
        self.gamma = gamma
        self.cache = None
        assert not (time_limit_ms is not None and iteration_limit is not None), "Cannot have both a time limit and an iteration limit"
        self.time_limit_ms = time_limit_ms
        self.iteration_limit = iteration_limit
        self.n_adversaries = n_adversaries
        match policy:
            case "ucb":
                self.policy = self._ucb1_policy
            case "random":
                self.policy = self._random_policy
            case _:
                raise ValueError("Invalid policy")

    def search(self, state: State, use_cached_tree: bool = True):
        if use_cached_tree:
            root = self._get_cached_tree(state)
        else:
            root = Node.root(state)
        if self.time_limit_ms is not None:
            deadline = time.time() + self.time_limit_ms / 1000
            while time.time() < deadline:
                self._train(root)
        elif self.iteration_limit is not None:
            for _ in range(self.iteration_limit):
                self._train(root)
        else:
            raise ValueError("Must have either a time limit or an iteration limit")

        print(f"Root node has been visited {root.num_visits} times.")
        for child in root.children:
            print(f"Action: {int(child.action[0])}, Value: {child.avg_value:.5f}")
        self.cache = root
        return root.best_action

    def _get_cached_tree(self, state: State):
        if self.cache is None:
            return Node.root(state)
        child = self.cache.get_child_with_state(state, self.n_adversaries + 1)
        if child is None:
            return Node.root(state)
        return child

    def _train(self, root: Node):
        node = self._select(root)
        if node.num_visits > 0:
            node = self._expand(node)
        value = self._simulation(node)
        node.backpropate(value, self.gamma)

    def _ucb1_policy(self, node: Node):
        best_child = node.children[0]
        best_value = best_child.ucb1(self.exploration_constant)
        for child in node.children[1:]:
            node_value = child.ucb1(self.exploration_constant)
            if node_value > best_value:
                best_value = node_value
                best_child = child
        return best_child

    def _random_policy(self, node: Node):
        return random.choice(node.children)

    def _select(self, node: Node) -> Node:
        """Select a move that seems promising accoring to the policy."""
        while not node.is_leaf:
            node = self.policy(node)
        return node

    def _expand(self, node: Node) -> Node:
        """Expand a leaf node by adding all its children"""
        assert node.is_leaf
        self.env.set_state(node.state)
        next_player = (node.current_player + 1) % (self.n_adversaries + 1)
        for action in self.env.available_joint_actions():
            self.env.set_state(node.state)
            step = self.env.step(action)
            child = Node(
                step.state,
                action,
                node,
                step.is_terminal,
                next_player,
                step.reward,
            )
            node.children.append(child)
        if len(node.children) == 0:
            return node
        return self._random_policy(node)

    def _simulation(self, node: Node) -> float:
        if node.is_terminal:
            return node.value

        self.env.set_state(node.state)
        done = False
        total_reward = 0.0
        current_gamma = 1.0
        turn = node.current_player
        n_adversarial_players = self.n_adversaries + 1
        while not done:
            action = self.env.action_space.sample(self.env.available_actions())
            step = self.env.step(action)
            done = step.done
            # We assume that the maximizing player is always the first one
            # because the search initializes the root node with the maximizing player (0).
            if turn == 0:
                total_reward += step.reward * current_gamma
            else:
                total_reward -= step.reward * current_gamma
            current_gamma *= self.gamma
            turn = (turn + 1) % n_adversarial_players
        return total_reward

    def _backpropagate(self, node: Node, value: float):
        current_node = node
        while current_node is not None:
            current_node.total_value += value
            current_node.num_visits += 1
            current_node = current_node.parent
            value *= self.gamma
