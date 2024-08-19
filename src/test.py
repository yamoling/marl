from typing import Any, Literal, Self

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from icecream import ic
from lle import LLE
from rlenv import Episode, Transition
import rlenv
from torch import device

from lle import ObservationType
import marl
from marl.utils import Schedule
from marl.algo.qlearning.maic import MAICParameters
from marl.other.subgoal import LocalGraph

np.random.seed(0)
g = LocalGraph()


class LocalGraphTrainer(marl.Trainer):
    def __init__(self, update_type: Literal["step", "episode"], update_interval: int):
        super().__init__(update_type, update_interval)
        self.graph = LocalGraph()

    def update_episode(self, episode: Episode, episode_num: int, time_step: int) -> dict[str, Any]:
        self.graph.add_trajectory(episode.states.tolist())
        import matplotlib.pyplot as plt
        import networkx as nx

        pos = nx.planar_layout(self.graph.graph)
        nx.draw(self.graph.graph, pos=pos, with_labels=True)
        labels = nx.get_edge_attributes(self.graph.graph, "weight")
        nx.draw_networkx_edge_labels(self.graph.graph, pos, edge_labels=labels)
        plt.draw()
        plt.show()
        return {}

    def randomize(self):
        return

    def to(self, device: device) -> Self:
        return self


n_steps = 300_000
walkable_lasers = True
temperature = 10
# env = LLE.from_file("maps/lvl3_without_gem").obs_type(ObservationType.LAYERED).walkable_lasers(walkable_lasers).build()
env = LLE.level(3).obs_type(ObservationType.STATE).walkable_lasers(walkable_lasers).build()
env = rlenv.Builder(env).agent_id().time_limit(78, add_extra=True).build()


# ac_network = marl.nn.model_bank.CNN_ActorCritic.from_env(env)
ac_network = marl.nn.model_bank.SimpleActorCritic.from_env(env)
# ac_network.temperature = temperature

entropy_schedule = Schedule.linear(0.05, 0.001, round(2 / 3 * n_steps))
temperature_schedule = Schedule.linear(50, 1, round(2 / 3 * n_steps))

# ac_network = marl.nn.model_bank.Clipped_CNN_ActorCritic.from_env(env)
memory = marl.models.TransitionMemory(20)

logits_clip_low = -2.0
logits_clip_high = 2.0

trainer = marl.training.PPOTrainer(
    network=ac_network,
    memory=memory,
    gamma=0.99,
    batch_size=2,
    lr_critic=1e-4,
    lr_actor=1e-4,
    optimiser="adam",
    train_every="step",
    update_interval=8,
    n_epochs=4,
    clip_eps=0.2,
    c1=0.5,
    c2=0.01,
    c2_schedule=entropy_schedule,
    softmax_temp_schedule=temperature_schedule,
    logits_clip_low=logits_clip_low,
    logits_clip_high=logits_clip_high,
)

algo = marl.algo.PPO(
    ac_network=ac_network,
    train_policy=marl.policy.CategoricalPolicy(),
    #     train_policy=marl.policy.EpsilonGreedy.linear(
    #     1.0,
    #     0.05,
    #     n_steps=300_000,
    # ),
    test_policy=marl.policy.ArgMax(),
    # extra_policy=marl.policy.ExtraPolicy(env.n_agents),
    # extra_policy_every=50,
    logits_clip_low=logits_clip_low,
    logits_clip_high=logits_clip_high,
)

# logdir = f"logs/PPO-{env.name}-batch_{trainer.update_interval}_{trainer.batch_size}-gamma_{trainer.gamma}-WL_{walkable_lasers}-C2_{trainer.c2}-C1_{trainer.c1}"
# logdir += "-epsGreedy" if isinstance(algo.train_policy, marl.policy.EpsilonGreedy) else ""
# logdir += "-clipped" if isinstance(ac_network, marl.nn.model_bank.Clipped_CNN_ActorCritic) else ""
runner = marl.Runner(env, algo, trainer)
runner.run(logdir="logs/debug")

exit()

trajectory1 = np.random.choice([0, 2, 4], size=50).tolist()
trajectory2 = np.random.choice([1, 3, 5], size=50).tolist()

bottleneck = [trajectory1[-1], 5, trajectory2[0]]
ic(bottleneck)
g.add_trajectory(trajectory1 + [5] + trajectory2)

pos = nx.planar_layout(g.graph)
nx.draw(g.graph, pos=pos, with_labels=True)
labels = nx.get_edge_attributes(g.graph, "weight")
nx.draw_networkx_edge_labels(g.graph, pos, edge_labels=labels)
plt.draw()
plt.show()

print(g.graph)


g.partition()
