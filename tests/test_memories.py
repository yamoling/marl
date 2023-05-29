import numpy as np
from rlenv import Transition
import marl
from .utils import MockEnv


def test_nstep_n_equals_end():
    gamma = 0.99
    memory = marl.models.replay_memory.NStepReturnMemory(10, 5, gamma)
    env = MockEnv(2)
    obs = env.reset()
    expected_nstep_reward_t0 = 0
    for i in range(5):
        obs.data[0] = i
        t = Transition(obs, np.array([1]), 1, i == 4, {}, obs, False)
        memory.add(t)
        expected_nstep_reward_t0 += t.reward * gamma ** i

    t = memory[0]
    assert t.reward == expected_nstep_reward_t0
    for i in range(5):
        t = memory[i]
        assert np.all(t.obs_.data[0] == 4)
    

def test_3steps():
    gamma = 0.99
    n = 3
    memory = marl.models.replay_memory.NStepReturnMemory(10, n, gamma)
    env = MockEnv(2)
    obs = env.reset()
    expected_nstep_reward_t0 = 0
    for i in range(5):
        t = Transition(obs, np.array([1]), 1, i == 4, {}, obs, False)
        memory.add(t)
        if i < n:
            expected_nstep_reward_t0 += t.reward * gamma ** i

    t = memory[0]
    assert t.reward == expected_nstep_reward_t0

def test_3steps_truncated():
    gamma = 0.99
    n = 3
    memory = marl.models.replay_memory.NStepReturnMemory(10, n, gamma)
    env = MockEnv(2)
    obs = env.reset()
    expected_nstep_reward_t0 = 0
    for i in range(5):
        t = Transition(obs, np.array([1]), 1, False, {}, obs, i == 4)
        memory.add(t)
        if i < n:
            expected_nstep_reward_t0 += t.reward * gamma ** i

    t = memory[0]
    assert t.reward == expected_nstep_reward_t0