from marl.algo import MTCS
from marl.env import ConnectN

# MCTS move computation time
PROCESS_TIME = 5


def main():
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


if __name__ == "__main__":
    main()
