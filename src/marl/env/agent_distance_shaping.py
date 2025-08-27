import networkx as nx
import numpy as np
from lle import LLE, World
from marlenv import MARLEnv, RLEnvWrapper, Space
from marlenv.wrappers import PotentialShaping
from typing import Literal
from dataclasses import dataclass
from marl.utils import path


@dataclass
class AgentDistanceShaping[A: Space](PotentialShaping[A]):
    scale: float
    aggregation: Literal["sum", "mean"]

    def __init__(self, env: MARLEnv[A] | RLEnvWrapper[A], reward_scale: float = 0.01, aggregation: Literal["sum", "mean"] = "sum"):
        if isinstance(env, RLEnvWrapper):
            lle = env.unwrapped
        else:
            lle = env
        assert isinstance(lle, LLE)
        graph = path.build_single_agent_graph(lle.world)
        self._distances = self._precompute_distances(graph, lle.world)
        super().__init__(env)
        self.scale = reward_scale
        self._world = lle.world
        self.aggregation = aggregation

    def _precompute_distances(self, graph: nx.Graph, world: World):
        matrix = np.zeros((world.height, world.width), dtype=np.float32)
        for x, y in graph.nodes:
            min_dist = float("inf")
            for exit_pos in world.exit_pos:
                dist = nx.shortest_path_length(graph, (x, y), exit_pos)
                if dist < min_dist:
                    min_dist = dist
            matrix[x, y] = min_dist
        return matrix

    def compute_potential(self):
        distances = list()
        for pos in self._world.agents_positions:
            distances.append(self._distances[pos])
        match self.aggregation:
            case "sum":
                return np.sum(distances)
            case "mean":
                return np.mean(distances)
        raise ValueError(f"Unknown aggregation method: {self.aggregation}")
