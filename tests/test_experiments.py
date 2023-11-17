from .utils import MockEnv
from marl.training import DQNTrainer
import marl
import pickle
import torch


def setup_experiment():
    env = MockEnv(4)
    qnetwork = marl.nn.model_bank.MLP.from_env(env).to("cpu")
    policy = marl.policy.EpsilonGreedy.linear(1, 0.01, n_steps=100_000)

    return marl.Experiment.create(
        "test",
        algo=marl.qlearning.DQN(qnetwork=qnetwork, train_policy=policy),
        trainer=DQNTrainer(qnetwork, policy, mixer=marl.qlearning.VDN(env.n_agents)),
        env=env,
        test_interval=100,
        n_steps=100,
    )


def test_unpickle_experiment_and_update_network():
    """
    Create an experiment and pickle it to a file.
    Then, unpickle it and run it to check that the optimiser indeed optimises the weights of the NN.
    """
    env = MockEnv(4)
    qnetwork = marl.nn.model_bank.MLP.from_env(env).to("cpu")
    policy = marl.policy.EpsilonGreedy.linear(1, 0.01, n_steps=100_000)

    exp = marl.Experiment.create(
        "test",
        algo=marl.qlearning.DQN(qnetwork=qnetwork, train_policy=policy),
        trainer=DQNTrainer(qnetwork, policy, mixer=marl.qlearning.VDN(env.n_agents)),
        env=env,
        test_interval=100,
        n_steps=100,
    )
    runner = exp.create_runner("csv", seed=0)
    runner.train(1)
    param_values = [p.data.clone() for p in qnetwork.parameters()]
    with open("test.pkl", "wb") as f:
        pickle.dump(exp, f)

    with open("test.pkl", "rb") as f:
        exp: marl.Experiment = pickle.load(f)

    # Check that the parameters values are identical after unpickling
    assert len(list(qnetwork.parameters())) == len(param_values)
    for param, initial_param in zip(qnetwork.parameters(), param_values):
        assert torch.equal(param, initial_param)

    runner = exp.create_runner("csv", seed=0)
    runner.train(1)

    # Check that the parameters values have been updated
    for param, initial_param in zip(qnetwork.parameters(), param_values):
        assert not torch.equal(param, initial_param)
