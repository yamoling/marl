import marl
from rlenv.wrappers import AgentId
import torch
from marl.models import QNetwork, Mixer, EpisodeMemory, TransitionMemory
from marl.training import DQNNodeTrainer, DQNTrainer
from marl.utils import get_device

import numpy as np
from rlenv import RLEnv, DiscreteActionSpace, Observation


class MatrixGame(RLEnv[DiscreteActionSpace]):
    """Single step matrix game used in QTRAN, Qatten and QPLEX papers."""

    N_AGENTS = 2
    UNIT_DIM = 1
    OBS_SHAPE = (1,)
    STATE_SIZE = UNIT_DIM * N_AGENTS

    QPLEX_PAYOFF_MATRIX = [
        [8.0, -12, -12],
        [-12, 0, 0],
        [-12, 0, 0],
    ]

    def __init__(self, payoff_matrix: list[list[float]]):
        action_names = [chr(ord("A") + i) for i in range(len(payoff_matrix[0]))]
        super().__init__(
            action_space=DiscreteActionSpace(2, len(payoff_matrix[0]), action_names),
            observation_shape=MatrixGame.OBS_SHAPE,
            state_shape=(MatrixGame.STATE_SIZE,),
        )
        self.current_step = 0
        self.payoffs = payoff_matrix

    def reset(self):
        self.current_step = 0
        return self.observation()

    def observation(self):
        return Observation(
            np.array([[self.current_step]] * MatrixGame.N_AGENTS, np.float32),
            self.available_actions(),
            self.get_state(),
        )

    def step(self, actions):
        self.current_step += 1
        return self.observation(), self.payoffs[actions[0]][actions[1]], True, False, {}

    def render(self, _mode):
        return

    def get_state(self):
        return np.zeros((MatrixGame.STATE_SIZE,), np.float32)

    def seed(self, *_):
        return


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


def test_mixer_matrix_game(mixer: Mixer, payoff_matrix: list[list[float]], expected: list[list[float]]):
    # env = AgentId(MatrixGame(payoff_matrix))
    env = AgentId(MatrixGame(payoff_matrix))
    qnetwork = QNetworkTest.from_env(env)
    env.reset()
    policy = marl.policy.EpsilonGreedy.constant(1.0)
    trainer = DQNTrainer(
        qnetwork=qnetwork,
        train_policy=policy,
        double_qlearning=True,
        memory=EpisodeMemory(500),
        batch_size=32,
        train_interval=(1, "episode"),
        mixer=mixer,
        target_updater=marl.training.HardUpdate(50),
        gamma=0.99,
        optimiser="adam",
        lr=5e-4,
    ).to(get_device())
    algo = marl.qlearning.DQN(qnetwork, policy)
    exp = marl.Experiment.create("logs/test", algo, trainer, env, 10_000, 10_000)
    runner = exp.create_runner(0)
    runner.train(0)

    qvalues = qnetwork.qvalues(env.reset())
    for a0 in range(3):
        for a1 in range(3):
            qs = torch.tensor([qvalues[0][a0], qvalues[1][a1]]).unsqueeze(0)
            s = torch.tensor(env.get_state(), dtype=torch.float32).unsqueeze(0)
            actions = torch.nn.functional.one_hot(torch.tensor([a0, a1]), 3).unsqueeze(0)
            res = mixer.forward(qs, s, actions, qvalues.unsqueeze(0)).detach().cpu().item()
            diff = res - expected[a0][a1]
            print(f"{a0}/{a1}: {res} vs {expected[a0][a1]}")


def test_qmix():
    """Matrix game used in QPLEX and QTRAN papers."""
    mixer = marl.qlearning.QMix(1, 2, embed_size=32)
    # Expected results shown in the QPLEX paper
    expected_qmix = [
        [-8.0, -8.0, -8.0],
        [-8.0, 0.0, 0.0],
        [-8.0, 0.0, 0.0],
    ]
    test_mixer_matrix_game(mixer, MatrixGame.QPLEX_PAYOFF_MATRIX, expected_qmix)


def test_qatten():
    mixer = marl.qlearning.Qatten(
        2,
        MatrixGame.STATE_SIZE,
        MatrixGame.UNIT_DIM,
    )
    # Expected results shown in the paper
    expected = [
        [-6.2, -4.9, -4.9],
        [-4.9, -3.5, -3.5],
        [-4.9, -3.5, -3.5],
    ]
    test_mixer_matrix_game(mixer, MatrixGame.QPLEX_PAYOFF_MATRIX, expected)


def test_vdn():
    mixer = marl.qlearning.VDN(2)
    # Expected results shown in the paper
    expected = [
        [-6.2, -4.9, -4.9],
        [-4.9, 3.6, 3.6],
        [-4.9, 3.6, 3.6],
    ]
    test_mixer_matrix_game(mixer, MatrixGame.QPLEX_PAYOFF_MATRIX, expected)


def test_qplex():
    mixer = marl.qlearning.QPlex(
        2,
        3,
        10,
        MatrixGame.STATE_SIZE,
        32,
    )
    # Expected results shown in the paper
    expected = [
        [8, -12.1, -12.1],
        [-12.2, 0, 0],
        [-12.1, 0, 0],
    ]
    test_mixer_matrix_game(mixer, MatrixGame.QPLEX_PAYOFF_MATRIX, expected)


test_qplex()
# test_qatten()
# test_qmix()
