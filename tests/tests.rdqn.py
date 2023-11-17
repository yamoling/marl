import marl
import torch
from .utils import MockEnv


def test_rdqn_twice_same_input():
    """
    Providing twice the same input to the RDQN should
    not yield the same output because the hidden state
    should have changed.
    """
    env = MockEnv(2)
    qnetwork = marl.nn.model_bank.RNNQMix.from_env(env)
    policy = marl.policy.EpsilonGreedy.linear(1, 0.01, n_steps=10_000)
    algo = marl.qlearning.RDQN(qnetwork, policy)

    obs = env.reset()
    qvalues1 = algo.compute_qvalues(obs)
    qvalues2 = algo.compute_qvalues(obs)
    assert not torch.equal(qvalues1, qvalues2)
