from lle import World, LLE, Action
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
env = marlenv.Builder(env).agent_id().available_actions_mask(mask).build()


def draw_graph(g: nx.Graph, bottleneck: set, labels: np.ndarray):
    # Now find boundary edges (edges connecting nodes of different clusters)
    pos = {x: x for x in g.nodes}
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


class SpectralTrainer(marl.Trainer):
    def __init__(self):
        super().__init__("both", 1)
        self.edge_weights = {}
        # Spectral clustering with 2 clusters
        self.sc = SpectralClustering(2, affinity="precomputed", n_init=100)

    def update_step(self, transition: Transition, time_step: int) -> dict[str, Any]:
        logs = {}
        hw = np.array([world.height, world.width], dtype=np.float32)

        start_pos = transition.obs.data[0][:2]
        end_pos = transition.obs_.data[0][:2]
        start_pos = start_pos * hw
        end_pos = end_pos * hw
        start_pos = tuple(start_pos.tolist())
        end_pos = tuple(end_pos.tolist())
        self.edge_weights[(start_pos, end_pos)] = self.edge_weights.get((start_pos, end_pos), 0) + 1
        return logs

    def update_episode(self, episode: Episode, episode_num: int, time_step: int):
        logs = {}
        local_graph = nx.Graph()
        edges = [(src, dest, v) for (src, dest), v in self.edge_weights.items()]
        self.edge_weights = {}
        local_graph.add_weighted_edges_from(edges)
        W = nx.adjacency_matrix(local_graph).todense()  # Adjacency/Weight matrix of the graph
        labels = self.sc.fit_predict(W)
        label_dict = {node: label for node, label in zip(local_graph.nodes, labels)}
        # Visualize the clusters

        bottleneck = set()
        for src, dst in local_graph.edges():
            if label_dict[src] != label_dict[dst]:  # If the nodes are in different clusters
                bottleneck.add((src, dst))
        print("Boundary edges:", bottleneck)
        draw_graph(local_graph, bottleneck, labels)
        self.extend_bottleneck(local_graph, bottleneck, labels)
        return logs

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
            draw_graph(local_graph, bottleneck, labels)
        return bottleneck

    def randomize(self):
        return

    def to(self, _):
        return self


trainer = SpectralTrainer()
exp = marl.Experiment.create("logs/test", trainer=trainer, n_steps=50_000, test_interval=0, env=env)
exp.run(0)
