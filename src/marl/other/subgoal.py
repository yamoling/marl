from typing import Hashable
from scipy.sparse import csr_array
import networkx as nx
import numpy as np
from icecream import ic

WEIGHT = "weight"


class LocalGraph[T: Hashable]:
    """
    Build a local graph (i.e. a graph that only represents a subgraph of the global graph) from trajectories.

    https://dl.acm.org/doi/10.1145/1102351.1102454
    """

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_trajectory(self, trajectory: list[T]):
        for i in range(len(trajectory) - 1):
            source = trajectory[i]
            target = trajectory[i + 1]
            if source == target:
                continue
            if not self.graph.has_node(source):
                self.graph.add_node(source)
            if not self.graph.has_node(target):
                self.graph.add_node(target)

            if not self.graph.has_edge(source, target):
                self.graph.add_edge(source, target, **{WEIGHT: 1})
            else:
                self.graph[source][target][WEIGHT] += 1

    def partition(self):
        # The paper discusses the case of undirected graphs, so we convert to an undirected graph.
        undirected = self.graph.to_undirected(reciprocal=False, as_view=True)
        L: csr_array = nx.normalized_laplacian_matrix(undirected)
        laplacian = L.toarray()

        # The value of the eigenvalues give an approximation of
        eigenvalues, eigenvectors = np.linalg.eig(laplacian)
        # The second smaller eigenvalue is a good approximation of normalized cut weight.
        eigvalue = eigenvalues[1]
        # The corresponding (i.e. second) eigenvector gives "labels" to each node.
        eigvector = eigenvectors[:, 1]
        ic(eigvalue, eigvector)
        # Order the nodes by their label
        indices = np.argsort(eigvector)
        nodes = list(undirected.nodes)
        sorted_nodes = [nodes[i] for i in indices]

        # Consider every possible cut according to the sorted nodes.
        for i in range(1, len(sorted_nodes)):
            source = sorted_nodes[:i]
            target = sorted_nodes[i:]
            # Compute the normalized cut weight (NCut) and retrieve the corresponding cut set.
            cut_set = list(nx.edge_boundary(undirected, source, target, data=WEIGHT))
            cut_weight = sum(weight for u, v, weight in cut_set)
            volume_s = nx.volume(undirected, source, weight=WEIGHT)
            volume_t = nx.volume(undirected, target, weight=WEIGHT)

            cs = cut_weight * ((1 / volume_s) + (1 / volume_t))
            cs2 = nx.normalized_cut_size(undirected, source, target, weight=WEIGHT)
            ic(source, target, cs, cs2)


def normalized_cut_weight[T](digraph: nx.DiGraph, source: list[T], target: list[T]) -> float:
    hit_ratio_st = nx.cut_size(digraph, source, target, weight=WEIGHT) / nx.volume(digraph, source)
    hit_ratio_ts = nx.cut_size(digraph, target, source, weight=WEIGHT) / nx.volume(digraph, target)
    return hit_ratio_st + hit_ratio_ts


def approximated_normalized_cut_weight[T](digraph: nx.DiGraph, source: list[T], target: list[T]) -> float:
    """
    Equation (2) for Approximated Normalized Cut Weight.
    """
    graph = digraph.to_undirected(reciprocal=False, as_view=True)
    cut_size_st = nx.cut_size(graph, source, target, weight=WEIGHT)
    cut_size_ts = nx.cut_size(graph, target, source, weight=WEIGHT)
    volume_s = nx.volume(graph, source)
    volume_t = nx.volume(graph, target)
    numerator = cut_size_st + cut_size_ts
    return numerator / (volume_s + cut_size_ts) + numerator / (volume_t + cut_size_st)
