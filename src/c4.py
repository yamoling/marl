"""Connect4 Monte Carlo main."""

from marlenv import MARLEnv, DiscreteActionSpace
import numpy as np

from marlenv import Observation, State, Step
from marl.env.connect4 import GameBoard, StepResult
from marl.algo import mcts

# MCTS move computation time
PROCESS_TIME = 5


class C4Env(MARLEnv[DiscreteActionSpace]):
    def __init__(self, width: int = 7, height: int = 6, n: int = 4):
        self.board = GameBoard(width, height, n)
        action_space = DiscreteActionSpace(1, self.board.width)
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
        self.board.board = state.data.copy()
        self.board.turn = int(state.extras[0])
        n_completed = np.count_nonzero(self.board.board, axis=0)
        self.board.n_items_in_column = n_completed

    def render(self):
        self.board.show()


def main():
    env = C4Env(width=6, height=7, n=4)
    board = env.board
    done = False
    while not done:
        env.render()
        if board.turn == -1:
            col = int(input(f"Enter the column (0 to {board.width-1}): "))
        else:
            print("Computer's turn, please wait...")
            # root = Node(parent=None, board=board.board, turn=montecarlo.symbol)
            # line, col = montecarlo.compute_move(root)
            col = mcts.search(env, iteration_limit=1000, n_adversaries=1)[0]
        done = env.step([col]).is_terminal
        print()
    env.render()
    print("The game is over")


if __name__ == "__main__":
    main()
