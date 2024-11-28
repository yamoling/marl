import random
import numpy as np
from marl.algo import MTCS
from marl.env import ConnectN
import marlenv
from lle import LLE, Action
from matplotlib import pyplot as plt

# MCTS move computation time
PROCESS_TIME = 5


def main_connectn():
    env = ConnectN(width=5, height=10, n=4)
    board = env.board
    mcts = MTCS(env, iteration_limit=1000, n_adversaries=1, policy="random")
    done = False
    while not done:
        env.render()
        if board.turn == -1:
            col = int(input(f"Enter the column (0 to {board.width-1}): "))
        else:
            print("Computer's turn, please wait...")
            # root = Node(parent=None, board=board.board, turn=montecarlo.symbol)
            # line, col = montecarlo.compute_move(root)
            col = mcts.search(env.get_state())[0]
        done = env.step([col]).is_terminal
        print()
    env.render()
    print("The game is over")


def main_lle():
    # ERREUR
    # Le problème semble être que le "set_state" de LLE ne fonctionne pas bien
    random.seed(0)
    np.random.seed(0)
    plt.ion()
    env = LLE.level(2).single_objective()
    env = marlenv.Builder(env).time_limit(30).build()
    mcts = MTCS(env, iteration_limit=1000, policy="ucb", gamma=0.95)
    done = False
    t = 0
    while not done:
        t += 1
        print(f"Step {t}")
        env.render()
        action = mcts.search(env.get_state(), use_cached_tree=True)
        assert mcts.cache is not None
        print(f"Best action: {[Action(int(a)) for a in action]}")
        for child in mcts.cache.children:
            print(f"\t{[Action(int(a)) for a in child.action]}, Value: {child.avg_value:.5f}")
        done = env.step(action).is_terminal


if __name__ == "__main__":
    # main_connectn()
    main_lle()
