import networkx as nx
from icecream import ic
from marl.other.subgoal import LocalGraph
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
g = LocalGraph()

trajectory1 = np.random.choice([0, 2, 4], size=50).tolist()
trajectory2 = np.random.choice([1, 3, 5], size=50).tolist()

bottleneck = [trajectory1[-1], 5, trajectory2[0]]
ic(bottleneck)
g.add_trajectory(trajectory1 + [5] + trajectory2)

pos = nx.planar_layout(g.graph)
nx.draw(g.graph, pos=pos, with_labels=True)
labels = nx.get_edge_attributes(g.graph, "weight")
nx.draw_networkx_edge_labels(g.graph, pos, edge_labels=labels)
plt.draw()
plt.show()

print(g.graph)


g.partition()
