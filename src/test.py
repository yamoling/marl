import random
import numpy as np
from marl.algo import MCTS
from marl.algo.mcts import AlphaZero
from marl.env import ConnectN
import marlenv
from lle import LLE, Action
from matplotlib import pyplot as plt

# MCTS move computation time
PROCESS_TIME = 5


def main_connectn():
    env = ConnectN(width=5, height=10, n=4)
    board = env.board
    mcts = MCTS(env, iteration_limit=1000, n_adversaries=1, policy="random")
    done = False
    while not done:
        env.render()
        if board.turn == -1:
            col = int(input(f"Enter the column (0 to {board.width-1}): "))
        else:
            print("Computer's turn, please wait...")
            # root = Node(parent=None, board=board.board, turn=montecarlo.symbol)
            # line, col = montecarlo.compute_move(root)
            col = mcts.train(env.get_state())[0]
        done = env.step([col]).is_terminal
        print()
    env.render()
    print("The game is over")


def main_lle():
    random.seed(0)
    np.random.seed(0)
    plt.ion()
    env = LLE.level(1).state_type("layered").single_objective()
    env = marlenv.Builder(env).time_limit(50).centralised().build()
    mcts = AlphaZero(env, gamma=0.95, n_search_iterations=100)
    mcts.train(1000)

    env = LLE.level(3).state_type("layered").single_objective()
    env = marlenv.Builder(env).time_limit(78).centralised().build()
    mcts = AlphaZero(env, gamma=0.95, n_search_iterations=200)
    mcts.train(2000)

    env = LLE.level(6).state_type("layered").single_objective()
    env = marlenv.Builder(env).time_limit(78).centralised().build()
    mcts = AlphaZero(env, gamma=0.95, n_search_iterations=200)
    mcts.train(10_000)


if __name__ == "__main__":
    # main_connectn()
    main_lle()
