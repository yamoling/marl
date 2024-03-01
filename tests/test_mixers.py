import marl
import torch
import numpy as np
from rlenv.wrappers import AgentId

from marl.models import QNetwork

from .envs import TwoSteps, TwoStepsState, MatrixGame


class QNetworkTest(QNetwork):
    def __init__(self, input_shape: tuple[int, ...], extras_shape: tuple[int, ...], output_shape: tuple[int, ...]):
        super().__init__(input_shape, extras_shape, output_shape)
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(input_shape[0] + extras_shape[0], 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_shape[0]),
        )

    def forward(self, obs, extras):
        if extras is not None:
            obs = torch.concatenate([obs, extras], dim=-1)
        return self.nn(obs)


def test_qmix_value():
    """
    Demonstration of QMix higher representation capabilities against VDN as described in the paper.

    https://arxiv.org/pdf/1803.11485.pdf
    Appendix B.
    """
    env = TwoSteps()
    env.reset()

    qnetwork = QNetworkTest.from_env(env)
    policy = marl.policy.EpsilonGreedy.constant(1.0)
    memory = marl.models.EpisodeMemory(500)
    mixer = marl.qlearning.QMix(env.state_shape[0], env.n_agents, embed_size=8)
    # mixer = marl.qlearning.VDN(2)
    device = marl.utils.get_device()
    trainer = marl.training.DQNNodeTrainer(
        qnetwork=qnetwork,
        train_policy=policy,
        double_qlearning=True,
        memory=memory,
        batch_size=32,
        train_interval=(1, "episode"),
        mixer=mixer,
        target_updater=marl.training.HardUpdate(100),
        gamma=0.99,
        optimiser="adam",
        lr=1e-4,
    )
    algo = marl.qlearning.DQN(qnetwork, policy)
    exp = marl.Experiment.create("logs/test", algo, trainer, env, 10_000, 10_000)
    runner = exp.create_runner(0)
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
        assert np.allclose(np.array(payoff_matrix), np.array(expected[state]), atol=0.1)


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

    qnetwork = QNetworkTest.from_env(env)
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
