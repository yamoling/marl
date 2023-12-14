from marl.training import nodes


v = nodes.ValueNode(10)
n1 = nodes.Add(v)
n2 = nodes.Add(v, n1)

n3 = nodes.Add(v)
n4 = nodes.Add(v, n3)

n5 = nodes.Add(n2, v)
n6 = nodes.Add(n4, n5)
n7 = nodes.Add(n6, v)
print(n7.value)
