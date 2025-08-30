from typing import Sequence
import numpy as np
from marlenv import MARLEnv, MultiDiscreteSpace, DiscreteSpace, Observation, State, Step
from dataclasses import dataclass


LEFT = 0
RIGHT = 1


@dataclass
class DeepSea(MARLEnv[MultiDiscreteSpace]):
    """
    Deep Sea environment to test for deep exploration.

    The agent explores a 2D grid where the bottom-right corner (max_depth, max_depth) is the goal and is the only state to yield a reward.
    The agent starts in the top-left corner (0, 0).
    The agent has two actions: left or right, and taking an action makes the agent dive one row deeper. The agent can not go beyond the grid boundaries.
    Going right gives a penalty of (0.01 / max_depth).
    """

    max_depth: int

    def __init__(self, depth: int):
        super().__init__(
            n_agents=1,
            action_space=DiscreteSpace(size=2, labels=["left", "right"]).repeat(1),
            observation_shape=(2,),
            state_shape=(2,),
        )
        self.max_depth = depth
        self._row = 0
        self._col = 0
        self._step_right_penalty = -0.01 / self.max_depth

    def get_observation(self) -> Observation:
        return Observation(np.array([self._row, self._col], dtype=np.float32), self.available_actions())

    def get_state(self) -> State:
        return State(np.array([self._row, self._col], dtype=np.float32))

    def reset(self):
        self._row = 0
        self._col = 0
        return self.get_observation(), self.get_state()

    def step(self, action: Sequence[int]):
        self._row += 1
        if action[0] == LEFT:
            self._col -= 1
        else:
            self._col += 1
        self._col = max(0, self._col)
        if action[0] == RIGHT:
            if self._row == self.max_depth:
                reward = 1.0
            else:
                reward = self._step_right_penalty
        else:
            reward = 0.0
        return Step(
            self.get_observation(),
            self.get_state(),
            reward,
            done=self._row == self.max_depth,
        )

    def set_state(self, state: State):
        self._row, self._col = state.data

    @property
    def agent_state_size(self):
        return 2
