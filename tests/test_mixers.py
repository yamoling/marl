import marl
import torch
import numpy as np
from rlenv.wrappers import AgentId, LastAction

from marl.models import EpisodeMemory
from marl.nn import model_bank
from marl.training import DQNTrainer

from .envs import TwoSteps, TwoStepsState, MatrixGame


def test_qmix_value():
    """
    Demonstration of QMix higher representation capabilities against VDN as described in the paper.

    https://arxiv.org/pdf/1803.11485.pdf
    Appendix B.
    """
    env = TwoSteps()
    mixer = marl.qlearning.QMix(env.state_shape[0], env.n_agents, embed_size=8)
    device = marl.utils.get_device()
    trainer = marl.training.DQNTrainer(
        qnetwork=model_bank.MLP.from_env(env, hidden_sizes=[64, 64]),
        train_policy=marl.policy.EpsilonGreedy.constant(1.0),
        double_qlearning=True,
        memory=marl.models.EpisodeMemory(500),
        batch_size=32,
        train_interval=(1, "episode"),
        mixer=mixer,
        target_updater=marl.training.HardUpdate(100),
        gamma=0.99,
        optimiser="adam",
        lr=1e-4,
    )
    algo = marl.qlearning.DQN(trainer.qnetwork, trainer.policy)
    runner = marl.Experiment.create("logs/test", algo, trainer, env, 10_000, 10_000).create_runner(0)
    runner.to(device)
    runner.train(0)

    # Expected results shown in the paper
    expected = {
        TwoStepsState.INITIAL: [[6.93, 6.93], [7.92, 7.92]],
        TwoStepsState.STATE_2A: [[7.0, 7.0], [7.0, 7.0]],
        TwoStepsState.STATE_2B: [[0, 1], [1, 8]],
    }

    for state in TwoStepsState.INITIAL, TwoStepsState.STATE_2A, TwoStepsState.STATE_2B:
        env.force_state(state)
        obs = env.observation()
        qvalues = algo.compute_qvalues(obs)
        payoff_matrix = [[0.0, 0.0], [0.0, 0.0]]
        for a0 in range(2):
            for a1 in range(2):
                qs = torch.tensor([qvalues[0][a0], qvalues[1][a1]]).unsqueeze(0).to(device)
                s = torch.tensor(obs.state, dtype=torch.float32).unsqueeze(0).to(device)
                res = mixer.forward(qs, s).detach()
                payoff_matrix[a0][a1] = res.item()
        assert np.allclose(np.array(payoff_matrix), np.array(expected[state]), atol=0.2)


def test_qplex():
    """
    Test QPlex against the matrix game shown in the paper.

    In the paper, the agent ID and the last actions are given as inputs to the network.
    """
    env = LastAction(AgentId(MatrixGame(MatrixGame.QPLEX_PAYOFF_MATRIX)))
    qnetwork = model_bank.RNN.from_env(env)
    policy = marl.policy.EpsilonGreedy.constant(1.0)
    mixer = marl.qlearning.mixers.QPlex(
        env.n_agents,
        env.n_actions,
        10,
        env.state_shape[0],
        64,
        # transformation=True,
    )
    trainer = DQNTrainer(
        qnetwork=qnetwork,
        train_policy=policy,
        double_qlearning=True,
        memory=EpisodeMemory(5000),
        batch_size=32,
        train_interval=(1, "episode"),
        mixer=mixer,
        target_updater=marl.training.HardUpdate(200),
        grad_norm_clipping=10,
        gamma=0.99,
        optimiser="rmsprop",
        lr=5e-4,
    )
    runner = marl.Experiment.create(
        "logs/test",
        marl.qlearning.DQN(qnetwork, policy),
        trainer,
        env,
        500,
        0,
    ).create_runner(0)
    device = marl.utils.get_device()
    runner.to(device)

    for _epoch in range(10):
        # Train the model for 500 time steps.
        # If it converged, the test passes.
        # If it did not converge, we train for an additional 500 time steps (at most 10 times).
        runner.train(0)

        # Then check if the learned Q-function matches the expected qvalues
        obs = env.reset()
        qnetwork.reset_hidden_states()
        qvalues = qnetwork.qvalues(obs).to(device)
        predicted = np.zeros((3, 3))
        for a0 in range(3):
            for a1 in range(3):
                qs = torch.tensor([qvalues[0][a0], qvalues[1][a1]]).unsqueeze(0).to(device)
                s = torch.tensor(env.get_state(), dtype=torch.float32).unsqueeze(0).to(device)
                actions = torch.nn.functional.one_hot(torch.tensor([a0, a1]), 3).unsqueeze(0).to(device)
                predicted[a0, a1] = mixer.forward(qs, s, actions, qvalues.unsqueeze(0)).detach().cpu().item()
        if np.allclose(predicted, MatrixGame.QPLEX_PAYOFF_MATRIX, atol=1):
            return
    assert False, "The QPLEX mixer did not converge to the expected values."


def test_mixers():
    """Matrix game used in QPLEX and QTRAN papers."""
    env = AgentId(
        MatrixGame(
            [
                [8, -12, -12],
                [-12, 0, 0],
                [-12, 0, 0],
            ]
        )
    )
    env.reset()

    qnetwork = model_bank.MLP.from_env(env)
    policy = marl.policy.EpsilonGreedy.constant(1.0)
    memory = marl.models.TransitionMemory(500)
    mixer = marl.qlearning.QMix(env.state_shape[0], env.n_agents, embed_size=8)
    # mixer = marl.qlearning.VDN(2)
    device = marl.utils.get_device()
    trainer = marl.training.DQNNodeTrainer(
        qnetwork=qnetwork,
        train_policy=policy,
        double_qlearning=True,
        memory=memory,
        batch_size=32,
        train_interval=(1, "step"),
        mixer=mixer,
        target_updater=marl.training.HardUpdate(50),
        gamma=0.99,
        optimiser="adam",
        lr=1e-4,
    )
    algo = marl.qlearning.DQN(qnetwork, policy)
    exp = marl.Experiment.create("logs/test", algo, trainer, env, 2000, 10_000)
    runner = exp.create_runner(0)
    runner.to(device)
    runner.train(0)

    # Expected results shown in the paper
    expected_qatten = [
        [-6.2, -4.9, -4.9],
        [-4.9, -3.5, -3.5],
        [-4.9, -3.5, -3.5],
    ]
    qvalues = qnetwork.qvalues(env.reset())
    for a0 in range(3):
        for a1 in range(3):
            qs = torch.tensor([qvalues[0][a0], qvalues[1][a1]]).unsqueeze(0).to(device)
            s = torch.tensor(env.get_state(), dtype=torch.float32).unsqueeze(0).to(device)
            res = mixer.forward(qs, s).detach().cpu().item()
            print(f"{a0}/{a1}: {res} vs {expected_qatten[a0][a1]}")
            diff = res - expected_qatten[a0][a1]
