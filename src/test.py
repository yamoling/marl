from typing import Any, Literal, Self
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from icecream import ic
from lle import LLE
from rlenv import Episode, Transition
import rlenv
import time
from torch import device

from lle import World
import marl
from marl.other.subgoal import LocalGraph


class LocalGraphTrainer(marl.Trainer):
    def __init__(
        self,
        world: World,
        update_type: Literal["step", "episode"],
        update_interval: int,
        trajectory_length: int,
        t_o: int,
        t_p: float,
    ):
        super().__init__(update_type, update_interval)
        self.graph = LocalGraph(t_o, t_p)
        self.world = world
        self.start_pos = tuple(world.start_pos[0])
        self.trajectory = [self.start_pos]
        self.trajectory_length = trajectory_length

    def update_step(self, transition: Transition, time_step: int) -> dict[str, Any]:
        agents_pos = tuple(self.world.agents_positions)
        self.trajectory.append(agents_pos[0])
        if transition.is_terminal or (time_step > 0 and time_step % self.trajectory_length == 0):
            self.graph.add_trajectory(self.trajectory)
            self.trajectory = [self.start_pos]
        if time_step > 0 and time_step % self.trajectory_length == 0:
            self.graph.partition()
            self.graph.clear()
        return {}

    def update_episode(self, episode: Episode, episode_num: int, time_step: int) -> dict[str, Any]:
        return {}

    def randomize(self):
        return

    def to(self, device: device) -> Self:
        return self


def main():
    map_str = """
.  . . . . . . . . . @ . . . . . . . . . .
.  . . . . . . . . . @ . . . . . . . . . .
.  . . . . . . . . . @ . . . . . . . . . .
.  . . . . . . . . . @ . . . . . . . . . .
.  . . . . . . . . . @ . . . . . . . . . .
S0 . . . . . . . . . . . . . . . . . . . .
.  . . . . . . . . . @ . . . . . . . . . .
.  . . . . . . . . . @ . . . . . . . . . .
.  . . . . . . . . . @ . . . . . . . . . .
.  . . . . . . . . . @ . . . . . . . . . .
.  . . . . . . . . . @ . . . . . . . . . X
"""
    map_str2 = """
S0 . @ X
.  . . .
"""
    env = LLE.from_str(map_str).build()
    world = env.world
    # env = rlenv.Builder(env).time_limit(100).agent_id().build()

    trainer = LocalGraphTrainer(
        world,
        "episode",
        1,
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
