import marl
import lle
from lle import WorldState
from itertools import permutations


def random_initial_states(env: lle.LLE):
    # Three different spawn areas:
    # - below the bottom laser (exclude exit tiles)
    # - below the top laser
    # - above the top laser
    res = dict[int, list[WorldState]]()
    area0 = [(i, j) for i in range(7, env.height) for j in range(7)]  # Bot left rectangle
    area0 += [(i, j) for i in range(9, env.height) for j in range(7, env.width) if (i, j) not in env.world.exit_pos]  # Bot right rectangle
    # All combinations of the area for four agents (without replacement)
    res[0] = [WorldState(list(p), [False] * env.world.n_gems) for p in permutations(area0, env.n_agents)]

    area1 = [(5, j) for j in range(env.width)]
    res[300_000] = [WorldState(list(p), [False] * env.world.n_gems) for p in permutations(area1, env.n_agents)]

    area2 = [(i, j) for i in range(4) for j in range(2, env.width)]
    res[600_000] = [WorldState(list(p), [False] * env.world.n_gems) for p in permutations(area2, env.n_agents)]

    return marl.env.lle_curriculum.RandomInitialStates(env)


def create_lle():
    env = lle.LLE.level(6, lle.ObservationType.LAYERED, multi_objective=True)
    # env = curriculum(env, n_steps)
    env = marl.env.lle_curriculum.RandomInitialStates(env, accumulate=True)
    # from marl.env import ExtraObjective
    go_next = ""
    while go_next != "n":
        env.t = 0
        env.reset()
        env.render("human")
        print(f"t={env.t}")
        env.render("human")
        go_next = input("Press 'n' to go next")

    go_next = ""
    while go_next != "n":
        env.t = 300_000
        env.reset()
        env.render("human")
        print(f"t={env.t}")
        env.render("human")
        go_next = input("Press 'n' to go next")

    go_next = ""
    while go_next != "n":
        env.t = 600_000
        env.reset()
        env.render("human")
        print(f"t={env.t}")
        env.render("human")
        go_next = input("Press 'n' to go next")


if __name__ == "__main__":
    create_lle()
