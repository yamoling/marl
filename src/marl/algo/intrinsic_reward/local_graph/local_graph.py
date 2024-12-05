import networkx as nx
import numpy as np
from lle import World, WorldState
from marlenv import Transition
from sklearn.cluster import SpectralClustering

from marl.models.trainer import Trainer


class LocalGraphBottleneckFinder[T]:
    """
    Find bottlenecks via local graph building and spectral clustering.
    https://dl.acm.org/doi/10.1145/1102351.1102454
    """

    def __init__(self, n_apparition_threshold: int = 10, hit_ratio_threshold: float = 0.5):
        self.edge_occurrences = dict[tuple[T, T], int]()
        """Map each edge to its weight (int)"""
        self.t_o = n_apparition_threshold
        """Threshold for the minimum number of apparitions of a state before considering it as a potential bottleneck"""
        self.t_h = hit_ratio_threshold
        """Threshold for the minimum hit ratio of a state before considering it as a bottleneck"""
        self.hit_count = dict[tuple[T, T], int]()
        """Number of times an edge was classified as a bottleneck"""
        self.apparition_count = dict[tuple[T, T], int]()
        """Number of times an edge was encountered in the construction of the local graph"""

    def is_bottleneck(self, edge: tuple[T, T]):
        ratio = self.predict(edge)
        return ratio >= self.t_h

    def predict(self, edge: tuple[T, T]):
        n = self.apparition_count.get(edge, 0)
        # if n < self.t_o:
        #     return 0.0
        return self.hit_count.get(edge, 0) / n

    def _build_local_graph(self, states: list[T]):
        local_graph = nx.Graph()
        for i in range(len(states) - 1):
            prev_state = states[i]
            next_state = states[i + 1]
            edge = prev_state, next_state
            weight = self.edge_occurrences.get(edge, 0) + 1
            self.edge_occurrences[edge] = weight
            local_graph.add_edge(prev_state, next_state, weight=weight)
        # local_graph.add_weighted_edges_from((src, dest, v) for (src, dest), v in self.edge_occurrences.items())
        return local_graph

    def compute_bottleneck(self, states: list[T]):
        local_graph = self._build_local_graph(states)
        # print(f"Graph size: {len(local_graph.nodes)} nodes, {len(local_graph.edges)} edges")
        W = nx.adjacency_matrix(local_graph).todense()  # Adjacency/Weight matrix of the graph
        sc = SpectralClustering(2, affinity="precomputed", n_init=100)
        labels = sc.fit_predict(W)
        label_dict = {node: label for node, label in zip(local_graph.nodes, labels)}

        bottleneck = set()
        for src, dst in local_graph.edges():
            if label_dict[src] != label_dict[dst]:  # If the nodes are in different clusters
                bottleneck.add((src, dst))
        self.extend_bottleneck(local_graph, bottleneck, labels)
        # draw_graph(local_graph, bottleneck, labels)
        for edge in local_graph.edges:
            self.apparition_count[edge] = self.apparition_count.get(edge, 0) + 1
            if edge in bottleneck:
                self.hit_count[edge] = self.hit_count.get(edge, 0) + 1
        return bottleneck

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


class LocalGraphTrainer(Trainer):
    def __init__(self, local_graph: LocalGraphBottleneckFinder[WorldState], world: World):
        super().__init__("both")
        self.local_graph = local_graph
        self.world = world
        self.states = list[WorldState]()

    def update_step(self, transition: Transition, time_step: int):
        self.states.append(self.world.get_state())
        return {}

    def update_episode(self, episode, episode_num, time_step):
        self.local_graph.compute_bottleneck(self.states)
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
