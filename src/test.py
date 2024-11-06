import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from icecream import ic
from lle import LLE, Action
import marlenv

import marl
from marl.other.local_graph import LocalGraphBottleneckFinder


def main():
    env = LLE.from_file("maps/subgraph-1agent.toml").single_objective()
    env.render("human")
    world = env.world

    mask = np.ones((env.n_agents, env.n_actions), dtype=bool)
    mask[:, Action.STAY.value] = False
    env = marlenv.Builder(env).agent_id().available_actions_mask(mask).build()

    trainer = LocalGraphTrainer(
        world,
        t_o=10,
        t_p=0.25,
        trajectory_length=500,
    )
    exp = marl.Experiment.create("logs/test", trainer=trainer, n_steps=50_000, test_interval=0, env=env)
    exp.run(0)

    heatmap = np.zeros((world.height, world.width))
    for i in range(world.height):
        for j in range(world.width):
            pos = (i, j)
            n_hits = trainer.graph.hits.get(pos, 0)
            hit_percentage = n_hits / trainer.graph.node_apparition_count.get(pos, 1)
            heatmap[i, j] = hit_percentage
            ic(pos, n_hits, hit_percentage)
    sns.heatmap(heatmap, annot=True, fmt=".2f")
    plt.show()


main()
