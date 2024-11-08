from lle import LLE, Action, WorldState
import marl
import marlenv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from marl.other.local_graph import LocalGraphTrainer, LocalGraphBottleneckFinder


def get_bottlenecks_stats(finder: LocalGraphBottleneckFinder[WorldState]):
    """Project the edges on the 2D plane"""
    vertex_bottleneck_scores = dict[tuple[int, int], float]()
    edge_bottleneck_scores = dict[tuple[WorldState, WorldState], float]()
    for (start, end), hit_count in finder.hit_count.items():
        edge_count = finder.edge_occurrences[(start, end)]
        edge_bottleneck_scores[(start, end)] = hit_count / edge_count
        ratio = hit_count / edge_count
        vertices = start.agents_positions + end.agents_positions
        for vertex in vertices:
            vertex_bottleneck_scores[vertex] = ratio
    return vertex_bottleneck_scores, edge_bottleneck_scores


def compute_heatmap(finder: LocalGraphBottleneckFinder[WorldState]):
    """Compute the heatmap of the bottlenecks"""
    vertex_scores, _ = get_bottlenecks_stats(finder)
    heatmap = np.zeros((world.height, world.width))
    for (i, j), score in vertex_scores.items():
        heatmap[i, j] = score
    return heatmap


map_name = "maps/subgraph-2agents.toml"
map_name = "maps/subgraph-1agent.toml"
env = LLE.from_file(map_name).obs_type("layered").single_objective()
world = env.world
if env.n_agents == 1:
    mask = np.full((env.n_agents, env.n_actions), True)
    mask[:, Action.STAY.value] = False
    env = marlenv.Builder(env).mask_actions(mask).build()

qnetwork = marl.nn.model_bank.CNN.from_env(env)
policy = marl.policy.EpsilonGreedy.linear(1.0, 0.05, 100_000)
algo = marl.algo.DQN(qnetwork, policy)
dqn_trainer = marl.training.DQNTrainer(
    qnetwork,
    policy,
    marl.models.TransitionMemory(10_000),
    mixer=marl.algo.VDN.from_env(env),
    lr=1e-4,
    gamma=0.95,
    train_interval=(100, "step"),
)


finder = LocalGraphBottleneckFinder()
trainer = LocalGraphTrainer(finder, world, None)
exp = marl.Experiment.create(logdir="logs/test", trainer=trainer, n_steps=100_000, test_interval=0, env=env)
exp.run()

scores_heatmap = np.zeros((world.height, world.width), dtype=np.float32)
scores_heatmap += compute_heatmap(finder)
sns.heatmap(scores_heatmap)
plt.show()
