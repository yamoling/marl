import numpy as np
import marl


def test_softmax():
    p = marl.policy.SoftmaxPolicy(3, 1.)
    available_actions = np.array([
        [1, 1, 0],
        [0, 0, 1]
    ])
    for _ in range(1_000):
        qvalues = np.random.rand(2, 3)
        action = p.get_action(qvalues, available_actions)
        assert 0 <= action[0] <= 1 and action[1] == 2
