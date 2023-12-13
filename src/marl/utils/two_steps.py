from enum import IntEnum
from typing import Literal
import rlenv
import numpy as np
from rlenv.models import DiscreteActionSpace
from rlenv.models.observation import Observation

PAYOFF_INITIAL = np.array([[0, 0], [0, 0]])

PAYOFF_2A = np.array([[7, 7], [7, 7]])

PAYOFF_2B = np.array([[0, 1], [1, 8]])


class State(IntEnum):
    INITIAL = 0
    STATE_2A = 1
    STATE_2B = 2
    END = 3

    def one_hot(self):
        res = np.zeros((4,), dtype=np.float32)
        res[self.value] = 1
        return res


class TwoSteps(rlenv.RLEnv[DiscreteActionSpace]):
    def __init__(self):
        self.state = State.INITIAL
        self._identity = np.identity(2, dtype=np.float32)
        super().__init__(
            DiscreteActionSpace(2, 2),
            (self.state.one_hot().shape[0] + 2,),
            (self.state.one_hot().shape[0],),
        )

    def reset(self) -> Observation:
        self.state = State.INITIAL
        return self.observation()

    def step(self, actions: np.ndarray):
        match self.state:
            case State.INITIAL:
                # In the initial step, only agent 0's actions have an influence on the state
                payoffs = PAYOFF_INITIAL
                if actions[0] == 0:
                    self.state = State.STATE_2A
                elif actions[0] == 1:
                    self.state = State.STATE_2B
                else:
                    raise ValueError(f"Invalid action: {actions[0]}")
            case State.STATE_2A:
                payoffs = PAYOFF_2A
                self.state = State.END
            case State.STATE_2B:
                payoffs = PAYOFF_2B
                self.state = State.END
            case State.END:
                raise ValueError("Episode is already over")
        obs = self.observation()
        reward = float(payoffs[actions[0], actions[1]])
        done = self.state == State.END
        return obs, reward, done, False, {}

    def get_state(self):
        return self.state.one_hot()

    def observation(self):
        obs_data = np.array([self.state.one_hot(), self.state.one_hot()])
        extras = self._identity
        return Observation(obs_data, self.available_actions(), self.get_state(), extras)

    def render(self, mode: Literal["human", "rgb_array"] = "human"):
        print(self.state)

    def force_state(self, state: State):
        self.state = state

    def seed(self, seed_value: int):
        pass
