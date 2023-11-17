from .node import Node


def compute_positions(nodes: set[Node]):
    """Compute the positions of the nodes in the graph"""
    levels = list[list[Node]]()
    for node in nodes:
        while len(levels) <= node.level:
            levels.append([])
        levels[node.level].append(node)
    max_level_width = max([len(level) for level in levels])
    positions = dict[Node, tuple[float, float]]()
    for level, level_nodes in enumerate(levels):
        node_spacing = max_level_width / len(level_nodes)
        first_node_padding = (max_level_width - len(level_nodes)) / 2
        for i, node in enumerate(level_nodes):
            positions[node] = (first_node_padding + +i * node_spacing, len(levels) - level)
    return positions
