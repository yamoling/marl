import numpy as np
from rlenv import Transition
import marl
from .utils import MockEnv


def test_nstep_n_equals_end():
    gamma = 0.99
    memory = marl.models.NStepMemory(10, 5, gamma)
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
    memory = marl.models.NStepMemory(10, n, gamma)
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
    memory = marl.models.NStepMemory(10, n, gamma)
    env = MockEnv(2)
    obs = env.reset()
    expected_nstep_reward_t0 = 0
    for i in range(5):
        t = Transition(obs, np.array([1]), 1, False, {}, obs, i == 4)
        memory.add(t)
        if i < n:
            expected_nstep_reward_t0 += t.reward * gamma ** i

    for i in range(3):
        t = memory[i]
        assert t.reward == expected_nstep_reward_t0
        assert not t.done
        assert t.truncated == (i == 2)
   
    t3 = memory[3]
    assert t3.reward == 1.99
    assert not t3.done
    assert t3.truncated

    t4 = memory[4]
    assert t4.reward == 1
    assert not t4.done
    assert t4.truncated


def test_episode_shorter_than_n():
    gamma = 0.99
    n = 5
    memory = marl.models.NStepMemory(10, n, gamma)
    env = MockEnv(2)
    obs = env.reset()
    for i in range(4):
        t = Transition(obs, np.array([1]), 1, i == 3, {}, obs, False)
        memory.add(t)

    t0 = memory[0]
    assert abs(t0.reward - (1 + 0.99 + 0.99 **2 + 0.99 ** 3)) < 1e-6
    assert t0.done
    assert not t0.truncated

    t1 = memory[1]
    assert abs(t1.reward - (1 + 0.99 + 0.99 **2)) < 1e-6
    assert t1.done
    assert not t1.truncated
    
    t2 = memory[2]
    assert abs(t2.reward - (1 + 0.99)) < 1e-6
    assert t2.done
    assert not t2.truncated

    t3 = memory[3]
    assert t3.reward == 1
    assert t3.done
    assert not t3.truncated
