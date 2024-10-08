from typing import Any, Literal, Self
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from icecream import ic
from lle import LLE, Action
from marlenv import Episode, Transition
from torch import device
import marlenv

from lle import World
import marl
from marl.other.subgoal import LocalGraph


class LocalGraphTrainer(marl.Trainer):
    def __init__(
        self,
        world: World,
        trajectory_length: int,
        t_o: int,
        t_p: float,
    ):
        super().__init__()
        all_pos = set((i, j) for i in range(world.height) for j in range(world.width)) - set(world.wall_pos)
        self.graph = LocalGraph(t_o, t_p, all_pos)
        self.world = world
        self.trajectory = [tuple[int, int](world.start_pos[0])]
        self.trajectory_length = trajectory_length

    def update_step(self, transition: Transition, time_step: int) -> dict[str, Any]:
        # print(time_step)
        prev_pos = self.trajectory[-1]
        agent_pos = self.world.agents_positions[0]
        diff = abs(agent_pos[0] - prev_pos[0] + agent_pos[1] - prev_pos[1])
        if diff > 1:
            action = Action(transition.action[0])
            ic(time_step, agent_pos, prev_pos, action)
        self.trajectory.append(agent_pos)
        if transition.is_terminal or (time_step > 0 and time_step % self.trajectory_length == 0):
            self.graph.add_trajectory(self.trajectory)
            self.trajectory = [agent_pos]
        if time_step > 0 and time_step % self.trajectory_length == 0:
            # pos = {x: (x[1], -x[0]) for x in self.graph.local_graph.nodes}
            # self.graph.show(pos)
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
.  . . . .  . . . . .  @ . . . . . . . . . .
.  . . . .  . . . . .  @ . . . . . . . . . .
.  . . . .  . . . . .  @ . . . . . . . . . .
.  . . . .  . . . . .  @ . . . . . . . . . .
.  . . . .  . . . . .  @ . . . . . . . . . .
S0 . . . .  . . . . .  . . . . . . . . . . .
.  . . . .  . . . . .  @ . . . . . . . . . .
.  . . . .  . . . . .  @ . . . . . . . . . .
.  . . . .  . . . . .  @ . . . . . . . . . .
.  . . . .  . . . . .  @ . . . . . . . . . .
.  . . . .  . . . . .  @ . . . . . . . . . X
"""
    env = LLE.from_str(map_str).single_objective()
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
