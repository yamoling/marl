import numpy as np
from marlenv import MARLEnv, DiscreteSpace, Step, State, Observation

from .board import GameBoard, StepResult


class ConnectN(MARLEnv[DiscreteSpace]):
    def __init__(self, width: int = 7, height: int = 6, n: int = 4):
        self.board = GameBoard(width, height, n)
        action_space = DiscreteSpace(1, self.board.width)
        observation_shape = (self.board.height, self.board.width)
        state_shape = observation_shape
        super().__init__(action_space, observation_shape, state_shape)

    def reset(self):
        self.board.clear()
        return self.get_observation(), self.get_state()

    def step(self, actions: list[int]):
        match self.board.play(actions[0]):
            case StepResult.NOTHING:
                done = False
                reward = 0
            case StepResult.WIN:
                done = True
                reward = 1
            case StepResult.TIE:
                done = True
                reward = 0
        return Step(self.get_observation(), self.get_state(), reward, done, False)

    def available_actions(self):
        """Full columns are not available."""
        return np.expand_dims(self.board.valid_moves(), axis=0)

    def get_observation(self):
        return Observation(self.board.board.copy(), self.available_actions())

    def get_state(self):
        return State(self.board.board.copy(), np.array([self.board.turn]))

    def set_state(self, state: State):
        self.board.board = state.data.copy()  # type: ignore Currently a type error because of the unchecked shape
        self.board.turn = int(state.extras[0])
        n_completed = np.count_nonzero(self.board.board, axis=0)
        self.board.n_items_in_column = n_completed

    def render(self):
        self.board.show()
