import torch
from marl.nn.mixers.qplex import QPlex

m = QPlex(state_shape=(20,), n_agents=3, n_actions=5)
q = torch.randn(7, 3)
st = torch.randn(7, 20)
all_q = torch.randn(7, 3, 5)
a = torch.nn.functional.one_hot(torch.randint(0, 5, (7, 3)), num_classes=5).float()
av = torch.ones(7, 3, 5, dtype=torch.bool)
out = m(q, st, one_hot_actions=a, all_qvalues=all_q, available_actions=av)
print(out.shape)

q2 = torch.randn(4, 6, 3)
st2 = torch.randn(4, 6, 20)
all_q2 = torch.randn(4, 6, 3, 5)
a2 = torch.nn.functional.one_hot(torch.randint(0, 5, (4, 6, 3)), num_classes=5).float()
av2 = torch.ones(4, 6, 3, 5, dtype=torch.bool)
out2 = m(q2, st2, one_hot_actions=a2, all_qvalues=all_q2, available_actions=av2)
print(out2.shape)
