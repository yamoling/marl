from typing import Hashable
import networkx as nx
import numpy as np


WEIGHT = "weight"


class LocalGraph[T: Hashable]:
    """
    Build a local graph (i.e. a graph that only represents a subgraph of the global graph) from trajectories.

    https://dl.acm.org/doi/10.1145/1102351.1102454
    """

    def __init__(self, t_o: int = 10, t_p: float = 0.25):
        self.graph = nx.DiGraph()
        self.subgoals = list[T]()

        # parameters for equation (3) of the paper.
        self.hits = dict[T, int]()
        self.node_apparition_count = dict[T, int]()
        self.t_o = t_o  # threshold for the minimal number of a node has been visited before considering it as a subgoal.
        self.t_p = t_p  # threshold for the percentage of times a node has been classified as a subgoal.

    def add_trajectory(self, trajectory: list[T]):
        for i in range(len(trajectory) - 1):
            source = trajectory[i]
            target = trajectory[i + 1]
            if not self.graph.has_edge(source, target):
                self.graph.add_edge(source, target, **{WEIGHT: 1})
            else:
                self.graph[source][target][WEIGHT] += 1

    def partition(self) -> tuple[float, tuple[list[T], list[T]], T]:
        # According to the paper, we keep track of the number of times a node has been visited
        # in order to check against the t_o threshold.
        for node in self.graph.nodes:
            self.node_apparition_count[node] = self.node_apparition_count.get(node, 0) + 1

        # The paper discusses the case of undirected graphs, so we convert to an undirected graph.
        # undirected = self.graph.to_undirected(reciprocal=False, as_view=True)
        laplacian = nx.normalized_laplacian_matrix(self.graph).toarray()

        eigenvalues, eigenvectors = np.linalg.eig(laplacian)
        # The second smaller eigenvalue is a good approximation of normalized cut weight.
        # eigvalue = eigenvalues[1]
        # The corresponding (i.e. second) eigenvector gives "labels" to each node.
        eigvector = eigenvectors[:, 1]
        # Order the nodes by their label
        indices = np.argsort(eigvector)
        nodes = list(self.graph.nodes)
        sorted_nodes = [nodes[i] for i in indices]

        min_cut_weight = float("inf")
        min_cut = None
        cut_node = None
        # Consider every possible cut according to the sorted nodes and take the one with the smallest normalized cut weight.
        for i in range(1, len(sorted_nodes)):
            source = sorted_nodes[:i]
            target = sorted_nodes[i:]
            # Compute the normalized cut weight (NCut) and retrieve the corresponding cut set.
            cut_set = list(nx.edge_boundary(self.graph, source, target, data=WEIGHT))
            cut_weight = sum(weight for u, v, weight in cut_set)
            # TODO: optimize this by precomputing the cumulative volume from left to right.
            volume_s = nx.volume(self.graph, source, weight=WEIGHT)
            volume_t = nx.volume(self.graph, target, weight=WEIGHT)

            cs = cut_weight * ((1 / volume_s) + (1 / volume_t))
            # cs2 = nx.normalized_cut_size(undirected, source, target, weight=WEIGHT)
            # ic(source, target, cs, cs2)
            if cs < min_cut_weight:
                min_cut_weight = cs
                min_cut = (source, target)
                cut_node = sorted_nodes[i]

        assert cut_node is not None and min_cut is not None
        self.hits[cut_node] = self.hits.get(cut_node, 0) + 1
        return min_cut_weight, min_cut, cut_node

    def is_subgoal(self, node: T):
        if node not in self.hits:
            return False
        if self.hits[node] < self.t_o:
            return False
        hit_percentage = self.hits[node] / self.node_apparition_count[node]
        return hit_percentage >= self.t_p

    def clear(self):
        self.graph.clear()


def normalized_cut_weight[T](digraph: nx.DiGraph, source: list[T], target: list[T]) -> float:
    hit_ratio_st = nx.cut_size(digraph, source, target, weight=WEIGHT) / nx.volume(digraph, source)
    hit_ratio_ts = nx.cut_size(digraph, target, source, weight=WEIGHT) / nx.volume(digraph, target)
    return hit_ratio_st + hit_ratio_ts


def approximated_normalized_cut_weight[T](digraph: nx.DiGraph, source: list[T], target: list[T]) -> float:
    """
    Equation (2) for Approximated Normalized Cut Weight.
    """
    graph = digraph.to_undirected(as_view=True)
    cut_size_st = nx.cut_size(graph, source, target, weight=WEIGHT)
    cut_size_ts = nx.cut_size(graph, target, source, weight=WEIGHT)
    volume_s = nx.volume(graph, source)
    volume_t = nx.volume(graph, target)
    numerator = cut_size_st + cut_size_ts
    return numerator / (volume_s + cut_size_ts) + numerator / (volume_t + cut_size_st)
