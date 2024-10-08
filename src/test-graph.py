from itertools import product
from lle import World, LLE, WorldState
import marlenv
import marl
from marl.training import Trainer
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import SpectralClustering
from typing import Any
from marlenv import Transition

bottlenecked_map = """
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
bottlenecked_map_2agents = """
.  . . . .  . . . . .  @ . . . . . . . . . X
.  . . . .  . . . . .  @ . . . . . . . . . .
.  . . . .  . . . . .  @ . . . . . . . . . .
.  . . . .  . . . . .  @ . . . . . . . . . .
S1 . . . .  . . . . .  @ . . . . . . . . . .
S0 . . . .  . . . . .  . . . . . . . . . . .
.  . . . .  . . . . .  @ . . . . . . . . . .
.  . . . .  . . . . .  @ . . . . . . . . . .
.  . . . .  . . . . .  @ . . . . . . . . . .
.  . . . .  . . . . .  @ . . . . . . . . . .
.  . . . .  . . . . .  @ . . . . . . . . . X
"""


def draw_graph(g: nx.Graph, bottleneck: set, labels: np.ndarray):
    # Now find boundary edges (edges connecting nodes of different clusters)
    pos = {}
    for node in g.nodes:
        pos[node] = (node[1], -node[0])
    pos = nx.spring_layout(g)
    not_bottlenecks = set(g.edges) - bottleneck
    nodes = list(g.nodes)
    one_side = [node for (i, node) in enumerate(nodes) if labels[i] == 0]
    other_side = [node for (i, node) in enumerate(nodes) if labels[i] == 1]

    # Draw intra-cluster edges in one color (e.g., blue)
    nx.draw_networkx_edges(g, pos, edgelist=not_bottlenecks, edge_color="black", width=2)
    # Draw boundary edges (bottlenecks) in another color (e.g., red)
    nx.draw_networkx_edges(g, pos, edgelist=bottleneck, edge_color="red", width=2)
    # Draw nodes with different colors based on their cluster labels
    nx.draw_networkx_nodes(g, pos, nodelist=one_side, node_color="#AAAAFF")
    nx.draw_networkx_nodes(g, pos, nodelist=other_side, node_color="#FFAAAA")

    # Draw node labels
    # nx.draw_networkx_labels(g, pos)
    nx.draw(g, pos, node_color=labels)
    plt.show()


Edge = tuple[WorldState, WorldState]


class BottleneckFinder(marl.Trainer):
    def __init__(self, world: World, trainer: Trainer | None = None, n_apparition_threshold: int = 10, hit_ratio_threshold: float = 0.5):
        super().__init__("both", 1)
        self.world = world
        self.trainer = trainer
        self.initial_state = world.get_state()
        self.prev_state = self.initial_state
        self.edge_weights = dict[Edge, int]()
        """Map each edge (int) to its weight (int)"""
        self.t_o = n_apparition_threshold
        """Threshold for the minimum number of apparitions of a state before considering it as a potential bottleneck"""
        self.t_h = hit_ratio_threshold
        """Threshold for the minimum hit ratio of a state before considering it as a bottleneck"""
        self.hit_count = dict[Edge, int]()
        """Number of times an edge was classified as a bottleneck"""
        self.apparition_count = dict[Edge, int]()
        """Number of times an edge was encountered in the construction of the local graph"""
        self.last_train = 0

    def is_bottleneck(self, edge: Edge):
        ratio = self.predict(edge)
        return ratio >= self.t_h

    def predict(self, edge: Edge):
        n = self.apparition_count.get(edge, 0)
        if n < self.t_o:
            return 0.0
        return self.hit_count.get(edge, 0) / n

    def update_step(self, transition: Transition[float, np.ndarray, np.ndarray], time_step: int) -> dict[str, Any]:
        logs = {}
        if self.trainer is not None:
            logs |= self.trainer.update_step(transition, time_step)
        state = transition.obs.state.tolist()
        state_ = transition.obs_.state.tolist()
        for i in range(self.world.n_agents):
            state[2 * i] = round(state[2 * i] * self.world.height)
            state[2 * i + 1] = round(state[2 * i + 1] * self.world.width)
            state_[2 * i] = round(state_[2 * i] * self.world.height)
            state_[2 * i + 1] = round(state_[2 * i + 1] * self.world.width)
        prev_state = WorldState.from_array(state, self.world.n_agents, self.world.n_gems)
        next_state = WorldState.from_array(state_, self.world.n_agents, self.world.n_gems)
        edge = prev_state, next_state
        self.edge_weights[edge] = self.edge_weights.get(edge, 0) + 1
        if transition.is_terminal and time_step - self.last_train > 1000:
            self.train()
            self.last_train = self.last_train + 1000
        return logs

    def train(self):
        local_graph = nx.Graph()
        edges = [(src, dest, v) for (src, dest), v in self.edge_weights.items()]
        self.edge_weights = {}
        local_graph.add_weighted_edges_from(edges)
        W = nx.adjacency_matrix(local_graph).todense()  # Adjacency/Weight matrix of the graph
        sc = SpectralClustering(2, affinity="precomputed", n_init=100)
        labels = sc.fit_predict(W)
        label_dict = {node: label for node, label in zip(local_graph.nodes, labels)}

        bottleneck = set()
        for src, dst in local_graph.edges():
            if label_dict[src] != label_dict[dst]:  # If the nodes are in different clusters
                bottleneck.add((src, dst))
        self.extend_bottleneck(local_graph, bottleneck, labels)
        draw_graph(local_graph, bottleneck, labels)
        for edge in local_graph.edges:
            self.apparition_count[edge] = self.apparition_count.get(edge, 0) + 1
            if edge in bottleneck:
                self.hit_count[edge] = self.hit_count.get(edge, 0) + 1
        return {}

    @staticmethod
    def extend_bottleneck(local_graph: nx.Graph, bottleneck: set, labels):
        """
        For every edge of the bottleneck, follow the graph as long as there is a single path between the nodes
        connected to the ends of the bottleneck.
        """
        to_visit = set()
        for src, dst in bottleneck:
            to_visit.add(src)
            to_visit.add(dst)
        visited = set()
        while len(to_visit) > 0:
            node = to_visit.pop()
            visited.add(node)
            # We are looking for "chains" of nodes with a single path, i.e. there are only two edges: one in and one out
            neighbours = set(local_graph.neighbors(node))
            if len(neighbours) == 2:
                bottleneck.update(local_graph.edges(node))
                to_visit.update(neighbours - visited)
            # draw_graph(local_graph, bottleneck, labels)
        return bottleneck

    def randomize(self):
        return

    def to(self, _):
        return self


map_str2 = """
.  . L0S . .  .  .  
.  .  .  . .  .  .  
.  .  .  . .  .  X  
S0 .  .  . .  .  .  
.  .  .  . .  .  X  
S1 .  .  . .  .  .  
.  .  .  . . L1N .  
"""

import random
from lle.env import SOLLE


class RandomInitialPos(marlenv.wrappers.RLEnvWrapper):
    def __init__(self, env: SOLLE, min_i: int, max_i: int, min_j: int, max_j: int):
        super().__init__(env)
        self.min_i = min_i
        self.min_j = min_j
        self.world = env.world
        self.lle = env
        self.ALL_INITIAL_POS = list(product(range(min_i, max_i + 1), range(min_j, max_j + 1)))

    def reset(self):
        super().reset()
        state = self.world.get_state()
        state.agents_positions = random.sample(self.ALL_INITIAL_POS, k=self.n_agents)
        self.world.set_state(state)
        obs = self.lle.core.get_observation()
        # self.render("human")
        return obs

    def seed(self, seed_value: int):
        random.seed(seed_value)
        return super().seed(seed_value)


env = LLE.from_str(bottlenecked_map).obs_type("layered").single_objective()
world = env.world
env = RandomInitialPos(env, 0, world.height - 1, 0, world.width // 2 - 1)
# env = LLE.level(6).obs_type("state").single_objective()

env = marlenv.Builder(env).time_limit(80).build()

qnetwork = marl.nn.model_bank.CNN.from_env(env)
policy = marl.policy.EpsilonGreedy.linear(1.0, 0.05, 100_000)
algo = marl.algo.DQN(qnetwork, policy, marl.policy.ArgMax())
dqn_trainer = marl.training.DQNTrainer(
    qnetwork,
    policy,
    marl.models.TransitionMemory(10_000),
    None,
    gamma=0.95,
    train_interval=(5, "step"),
)
dqn_trainer = None

bottleneck_count = dict[Edge, int]()
for seed in range(50):
    trainer = BottleneckFinder(world, None, n_apparition_threshold=10)
    exp = marl.Experiment.create("logs/test", trainer=trainer, n_steps=100_000, test_interval=0, env=env)
    exp.run(seed)
    for edge, count in trainer.apparition_count.items():
        p = trainer.predict(edge)
        if p > 0.1:
            bottleneck_count[edge] = bottleneck_count.get(edge, 0) + 1
            continue
            # print(f"{edge} encountered {count} times and has a hit ratio of {p}")
            world.set_state(edge[0])
            img_start = world.get_image()
            world.set_state(edge[1])
            img_end = world.get_image()
            axes = plt.subplot(1, 2, 1)
            axes.imshow(img_start)
            axes = plt.subplot(1, 2, 2)
            axes.imshow(img_end)
            from time import time

            plt.savefig(f"imgs/{p * 100:.2f}-{time()}.png")
            plt.close()
            # plt.show()
print("from lle import WorldState")
print("bottleneck_count = ", end="")
print(bottleneck_count)
