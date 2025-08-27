from itertools import product
import networkx as nx
from lle import World


def build_single_agent_graph(world: World) -> nx.Graph:
    graph = nx.Graph()
    all_positions = set(product(range(world.height), range(world.height)))
    walkable_positions = all_positions - set(world.wall_pos)

    for x, y in walkable_positions:
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbour = (x + dx, y + dy)
            if neighbour in walkable_positions:
                graph.add_edge((x, y), neighbour)
    return graph
