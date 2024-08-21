import networkx as nx


g = nx.DiGraph()

g.add_edge(1, 2, weight=1)
g.add_edge(2, 1, weight=1)

g = g.to_undirected()
weight = g.get_edge_data(2, 1)["weight"]
print(weight)
