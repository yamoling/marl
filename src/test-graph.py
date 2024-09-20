from lle import World, LLE, Action, WorldState
import marlenv
import marl
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import SpectralClustering
from typing import Any
from marlenv import Episode, Transition

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

env = LLE.from_str(map_str).obs_type("state").single_objective()
world = env.world

mask = np.ones((env.n_agents, env.n_actions), dtype=bool)
mask[:, Action.STAY.value] = False
env = marlenv.Builder(env).available_actions_mask(mask).build()


def draw_graph(g: nx.Graph, bottleneck: set, labels: np.ndarray):
    # Now find boundary edges (edges connecting nodes of different clusters)
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


class BottleneckFinder(marl.Trainer):
    def __init__(self, world: World, n_apparition_threshold: int = 10, hit_ratio_threshold: float = 0.5):
        super().__init__("both", 1)
        # Spectral clustering with 2 clusters
        self.sc = SpectralClustering(2, affinity="precomputed", n_init=100)
        self.world = world
        self.prev_state = self.world.get_state()
        self.edge_weights = dict[tuple[WorldState, WorldState], int]()
        """Map each edge (int) to its weight (int)"""
        self.t_o = n_apparition_threshold
        """Threshold for the minimum number of apparitions of a state before considering it as a potential bottleneck"""
        self.t_h = hit_ratio_threshold
        """Threshold for the minimum hit ratio of a state before considering it as a bottleneck"""
        self.hit_count = dict[WorldState, int]()
        """Number of times an edge was classified as a bottleneck"""
        self.apparition_count = dict[WorldState, int]()
        """Number of times an edge was encountered in the construction of the local graph"""

    def is_bottleneck(self, state: WorldState):
        n = self.apparition_count.get(state, 0)
        if n < self.t_o:
            return False
        hit_ratio = self.hit_count.get(state, 0) / n
        return hit_ratio >= self.t_h

    def update_step(self, transition: Transition[float, np.ndarray, np.ndarray], time_step: int) -> dict[str, Any]:
        new_state = self.world.get_state()
        start_pos = self.prev_state
        end_pos = new_state
        self.edge_weights[(start_pos, end_pos)] = self.edge_weights.get((start_pos, end_pos), 0) + 1
        self.prev_state = new_state
        return {}

    def update_episode(self, episode: Episode, episode_num: int, time_step: int):
        local_graph = nx.Graph()
        edges = [(src, dest, v) for (src, dest), v in self.edge_weights.items()]
        self.edge_weights = {}
        local_graph.add_weighted_edges_from(edges)
        W = nx.adjacency_matrix(local_graph).todense()  # Adjacency/Weight matrix of the graph
        labels = self.sc.fit_predict(W)
        label_dict = {node: label for node, label in zip(local_graph.nodes, labels)}

        bottleneck = set()
        for src, dst in local_graph.edges():
            if label_dict[src] != label_dict[dst]:  # If the nodes are in different clusters
                bottleneck.add((src, dst))
        self.extend_bottleneck(local_graph, bottleneck, labels)
        draw_graph(local_graph, bottleneck, labels)
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


trainer = BottleneckFinder(world)
exp = marl.Experiment.create("logs/test", trainer=trainer, n_steps=50_000, test_interval=0, env=env)
exp.run(0)
