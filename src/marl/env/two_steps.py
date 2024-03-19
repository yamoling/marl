from enum import IntEnum
from typing import Literal
import rlenv
import numpy as np
import numpy.typing as npt
from rlenv.models import DiscreteActionSpace
from rlenv.models.observation import Observation

PAYOFF_INITIAL = [[0, 0], [0, 0]]

PAYOFF_2A = [[7, 7], [7, 7]]

PAYOFF_2B = [[0, 1], [1, 8]]


class TwoStepsState(IntEnum):
    INITIAL = 0
    STATE_2A = 1
    STATE_2B = 2
    END = 3

    def one_hot(self):
        res = np.zeros((4,), dtype=np.float32)
        res[self.value] = 1
        return res


class TwoSteps(rlenv.RLEnv[DiscreteActionSpace]):
    """
    Two-steps game used in QMix paper (https://arxiv.org/pdf/1803.11485.pdf, section 5)
    to demonstrate is superior representationability compared to VDN.
    """

    def __init__(self):
        self.state = TwoStepsState.INITIAL
        self._identity = np.identity(2, dtype=np.float32)
        super().__init__(
            DiscreteActionSpace(2, 2),
            observation_shape=(self.state.one_hot().shape[0] + 2,),
            state_shape=self.state.one_hot().shape,
        )

    def reset(self) -> Observation:
        self.state = TwoStepsState.INITIAL
        return self.observation()

    def step(self, actions: npt.NDArray[np.int32]):
        match self.state:
            case TwoStepsState.INITIAL:
                # In the initial step, only agent 0's actions have an influence on the state
                payoffs = PAYOFF_INITIAL
                if actions[0] == 0:
                    self.state = TwoStepsState.STATE_2A
                elif actions[0] == 1:
                    self.state = TwoStepsState.STATE_2B
                else:
                    raise ValueError(f"Invalid action: {actions[0]}")
            case TwoStepsState.STATE_2A:
                payoffs = PAYOFF_2A
                self.state = TwoStepsState.END
            case TwoStepsState.STATE_2B:
                payoffs = PAYOFF_2B
                self.state = TwoStepsState.END
            case TwoStepsState.END:
                raise ValueError("Episode is already over")
        obs = self.observation()
        reward = payoffs[actions[0]][actions[1]]
        done = self.state == TwoStepsState.END
        return obs, reward, done, False, {}

    def get_state(self):
        return self.state.one_hot()

    def observation(self):
        obs_data = np.array([self.state.one_hot(), self.state.one_hot()])
        extras = self._identity
        return Observation(obs_data, self.available_actions(), self.get_state(), extras)

    def render(self, mode: Literal["human", "rgb_array"] = "human"):
        print(self.state)

    def force_state(self, state: TwoStepsState):
        self.state = state

    def seed(self, seed):
        return
