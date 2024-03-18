import marl
import lle
from lle import WorldState

if __name__ == "__main__":
    env = lle.LLE.level(6)
    world = env.world
    env.reset()
    i_positions = list(range(world.height - 1, -1, -1))
    j_positions = [3] * len(i_positions)
    initial_states = list[WorldState]()
    for i, j in zip(i_positions, j_positions):
        start_positions = [(i, j + n) for n in range(world.n_agents)]
        initial_states.append(WorldState(start_positions, [False] * world.n_gems))

    interval = 1_000_000 // len(initial_states)
    env = marl.env.CurriculumLearning(env, initial_states, interval)
