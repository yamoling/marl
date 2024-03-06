import numpy as np
from rlenv import RLEnv, DiscreteActionSpace, Observation


class MatrixGame(RLEnv[DiscreteActionSpace]):
    """Single step matrix game used in QTRAN, Qatten and QPLEX papers."""

    N_AGENTS = 2
    UNIT_DIM = 1
    OBS_SHAPE = (1,)
    STATE_SIZE = UNIT_DIM * N_AGENTS

    QPLEX_PAYOFF_MATRIX = [
        [8.0, -12.0, -12.0],
        [-12.0, 0.0, 0.0],
        [-12.0, 0.0, 0.0],
    ]

    def __init__(self, payoff_matrix: list[list[float]]):
        action_names = [chr(ord("A") + i) for i in range(len(payoff_matrix[0]))]
        super().__init__(
            action_space=DiscreteActionSpace(2, len(payoff_matrix[0]), action_names),
            observation_shape=MatrixGame.OBS_SHAPE,
            state_shape=(MatrixGame.STATE_SIZE,),
        )
        self.current_step = 0
        self.payoffs = payoff_matrix

    def reset(self):
        self.current_step = 0
        return self.observation()

    def observation(self):
        return Observation(
            np.array([[self.current_step]] * MatrixGame.N_AGENTS, np.float32),
            self.available_actions(),
            self.get_state(),
        )

    def step(self, actions):
        self.current_step += 1
        return self.observation(), self.payoffs[actions[0]][actions[1]], True, False, {}

    def render(self, _mode):
        return

    def get_state(self):
        return np.zeros((MatrixGame.STATE_SIZE,), np.float32)

    def seed(self, *_):
        return
