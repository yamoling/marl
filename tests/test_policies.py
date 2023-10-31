import numpy as np
import marl
from .utils import almost_equal

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

def test_egreedy_update():
    p = marl.policy.EpsilonGreedy.linear(1, 0.1, 10)
    for i in range(1, 11):
        p.update(i)
        expected = 1 - ((1 - 0.1) * i) / 10
        assert almost_equal(p.epsilon.value, expected)
    assert almost_equal(p.epsilon.value, 0.1)