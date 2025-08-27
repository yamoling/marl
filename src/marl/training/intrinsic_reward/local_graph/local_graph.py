import networkx as nx
import numpy as np
from lle import World, WorldState
from marlenv import Transition
from sklearn.cluster import SpectralClustering
from dataclasses import dataclass
from marl.models.trainer import Trainer


@dataclass
class LocalGraphBottleneckFinder[T]:
    """
    Find bottlenecks via local graph building and spectral clustering.
    https://dl.acm.org/doi/10.1145/1102351.1102454
    """

    edge_occurrences: dict[tuple[T, T], int]
    """Map each edge to its weight (int)"""
    t_o: int
    """Threshold for the minimum number of apparitions of a state before considering it as a potential bottleneck"""
    t_h: float
    """Threshold for the minimum hit ratio of a state before considering it as a bottleneck"""
    hit_count: dict[tuple[T, T], int]
    """Number of times an edge was classified as a bottleneck"""
    apparition_count: dict[tuple[T, T], int]
    """Number of times an edge was encountered in the construction of the local graph"""

    def __init__(self, n_apparition_threshold: int = 10, hit_ratio_threshold: float = 0.5):
        self.edge_occurrences = dict[tuple[T, T], int]()
        self.t_o = n_apparition_threshold
        self.t_h = hit_ratio_threshold
        self.hit_count = dict[tuple[T, T], int]()
        self.apparition_count = dict[tuple[T, T], int]()
        self.local_graph = nx.Graph()

    def predict_all(self):
        predictions = {edge: self.predict(edge) for edge in self.apparition_count.keys()}
        by_vertex = {}
        for (src, dst), pred in predictions.items():
            by_vertex[src] = by_vertex.get(src, 0) + pred
            by_vertex[dst] = by_vertex.get(dst, 0) + pred
        return predictions, by_vertex

    def is_bottleneck(self, edge: tuple[T, T]):
        ratio = self.predict(edge)
        return ratio >= self.t_h

    def predict(self, edge: tuple[T, T]):
        n = self.apparition_count.get(edge, 0)
        # if n < self.t_o:
        #     return 0.0
        return self.hit_count.get(edge, 0) / n

    def add_trajectory(self, states: list[T]):
        for i in range(len(states) - 1):
            prev_state = states[i]
            next_state = states[i + 1]
            if prev_state == next_state:
                continue
            edge = prev_state, next_state
            weight = self.edge_occurrences.get(edge, 0) + 1
            self.edge_occurrences[edge] = weight
            self.local_graph.add_edge(prev_state, next_state, weight=weight)

    def find_bottleneck(self):
        """
        Compute the edges that form the bottleneck according to the current local graph.

        This method also updates the hit_count and apparition_count dictionaries that are used
        to determine if an edge is a bottleneck in the `predict` method.
        """
        W = nx.adjacency_matrix(self.local_graph).todense()  # Adjacency/Weight matrix of the graph
        sc = SpectralClustering(2, affinity="precomputed", n_init=100)
        labels = sc.fit_predict(W)
        label_dict = {node: label for node, label in zip(self.local_graph.nodes, labels)}

        bottleneck = set[tuple[T, T]]()
        for src, dst in self.local_graph.edges():
            if label_dict[src] != label_dict[dst]:  # If the nodes are in different clusters
                bottleneck.add((src, dst))
        self.extend_bottleneck(self.local_graph, bottleneck, labels)
        # draw_graph(local_graph, bottleneck, labels)
        for edge in self.local_graph.edges:
            self.apparition_count[edge] = self.apparition_count.get(edge, 0) + 1
            if edge in bottleneck:
                self.hit_count[edge] = self.hit_count.get(edge, 0) + 1
        return bottleneck, labels

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

    def clear(self):
        self.local_graph.clear()


class LocalGraphTrainer(Trainer):
    def __init__(self, local_graph: LocalGraphBottleneckFinder[WorldState], world: World):
        super().__init__()
        self.local_graph = local_graph
        self.world = world
        self.states = list[WorldState]()

    def update_step(self, transition: Transition, time_step: int):
        self.states.append(self.world.get_state())
        return {}

    def update_episode(self, episode, episode_num, time_step):
        self.local_graph.add_trajectory(self.states)
        self.local_graph.find_bottleneck()
        self.local_graph.clear()
        self.states.clear()
        return {}

    def to(self, _):
        return self

    def randomize(self):
        return


def draw_graph(g: nx.Graph, bottleneck: set, labels: np.ndarray):
    import matplotlib.pyplot as plt

    # Now find boundary edges (edges connecting nodes of different clusters)
    pos = {}
    for node in g.nodes:
        pos[node] = (node[1], -node[0])
    # pos = nx.spring_layout(g)
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
    return plt
