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


def test_rdqn_new_episode():
    env = MockEnv(2)
    qnetwork = marl.nn.model_bank.RNNQMix.from_env(env)
    policy = marl.policy.EpsilonGreedy.linear(1, 0.01, n_steps=10_000)
    algo = marl.qlearning.RDQN(qnetwork, policy)

    qvalues1 = algo.compute_qvalues(env.reset())
    algo.new_episode()
    qvalues2 = algo.compute_qvalues(env.reset())
    assert torch.equal(qvalues1, qvalues2)


def test_rdqn_from_train_to_test():
    env = MockEnv(2)
    qnetwork = marl.nn.model_bank.RNNQMix.from_env(env)
    policy = marl.policy.EpsilonGreedy.linear(1, 0.01, n_steps=10_000)
    algo = marl.qlearning.RDQN(qnetwork, policy)

    obs = env.reset()
    qvalues1 = algo.compute_qvalues(obs)
    _ = algo.compute_qvalues(obs)
    algo.set_testing()
    qvalues2 = algo.compute_qvalues(obs)
    assert torch.equal(qvalues1, qvalues2)
