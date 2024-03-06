import pickle
import marl
from rlenv.wrappers import AgentId, LastAction
from rlenv import Transition
from rlenv.models import EpisodeBuilder
import torch
from marl.models import QNetwork, Mixer, EpisodeMemory, TransitionMemory, NN, RecurrentQNetwork
from marl.training import DQNTrainer
from marl.utils import get_device
from marl.nn import model_bank

import numpy as np
from rlenv import RLEnv, DiscreteActionSpace, Observation


class MatrixGame(RLEnv[DiscreteActionSpace]):
    """Single step matrix game used in QTRAN, Qatten and QPLEX papers."""

    N_AGENTS = 2
    UNIT_DIM = 1
    OBS_SHAPE = (1,)
    STATE_SIZE = UNIT_DIM * N_AGENTS

    QPLEX_PAYOFF_MATRIX = [
        [8.0, -12.0, -12.0],
        [-12.0, 0.0, 0.0],
        [-12.0, 0.0, 0.0],
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


def show_matrix(env: RLEnv, qnetwork: QNetwork, mixer: Mixer):
    obs = env.reset()
    if isinstance(qnetwork, RecurrentQNetwork):
        qnetwork.reset_hidden_states()
    qvalues = qnetwork.qvalues(obs)
    matrix_to_print = "A0\\A1\t  0\t  1\t  2\n"
    for a0 in range(3):
        matrix_to_print += f"{a0}\t"
        for a1 in range(3):
            qs = torch.tensor([qvalues[0][a0], qvalues[1][a1]]).unsqueeze(0)
            s = torch.tensor(env.get_state(), dtype=torch.float32).unsqueeze(0)
            actions = torch.nn.functional.one_hot(torch.tensor([a0, a1]), 3).unsqueeze(0)
            res = mixer.forward(qs, s, actions, qvalues.unsqueeze(0)).detach().cpu().item()
            matrix_to_print += f"{res:.2f}\t"
        matrix_to_print += "\n"
    print(matrix_to_print)


def test_mixer_matrix_game(mixer: Mixer, payoff_matrix: list[list[float]], expected: list[list[float]]):
    env = LastAction(AgentId(MatrixGame(payoff_matrix)))
    # qnetwork = model_bank.MLP.from_env(env, [64, 64])
    qnetwork = model_bank.MLP.from_env(env)
    policy = marl.policy.EpsilonGreedy.constant(1.0)
    trainer = DQNTrainer(
        qnetwork=qnetwork,
        train_policy=policy,
        double_qlearning=True,
        memory=all_interactions(env),
        batch_size=32,
        train_interval=(1, "episode"),
        mixer=mixer,
        target_updater=marl.training.HardUpdate(200),
        grad_norm_clipping=10,
        gamma=0.99,
        optimiser="rmsprop",
        lr=5e-4,
    ).to(get_device())

    import torch

    def show(nn: NN):
        print(next(nn.parameters()))
        for i, p in enumerate(nn.parameters()):
            print(i, p.shape)

    torch.manual_seed(0)
    trainer.qnetwork.randomize()
    show(trainer.qnetwork)
    torch.manual_seed(0)
    trainer.qtarget.randomize()
    show(trainer.qtarget)
    torch.manual_seed(0)
    trainer.mixer.randomize()
    show(trainer.mixer)
    torch.manual_seed(0)
    trainer.target_mixer.randomize()
    show(trainer.target_mixer)
    # for group in trainer.optimiser.param_groups:
    #     for key, value in group.items():
    #         print(key, len(value))

    for t in range(10_000):
        if t % 500 == 0:
            show_matrix(env, qnetwork, mixer)
        trainer._update(t)
    exit()


def test_qplex():
    mixer = marl.qlearning.QPlex(
        2,
        3,
        10,
        MatrixGame.STATE_SIZE,
        64,
    )
    # Expected results shown in the paper
    expected = [
        [8, -12.1, -12.1],
        [-12.2, 0, 0],
        [-12.1, 0, 0],
    ]
    test_mixer_matrix_game(mixer, MatrixGame.QPLEX_PAYOFF_MATRIX, expected)


def all_interactions(env: RLEnv):
    memory = EpisodeMemory(32)

    actions_to_perform = [
        [0, 0],
        [1, 1],
        [1, 2],
        [2, 0],
        [2, 1],
        [2, 2],
        [0, 0],
        [0, 1],
        [0, 2],
        [1, 0],
        [1, 1],
        [1, 2],
        [2, 0],
        [2, 1],
        [2, 2],
        [0, 0],
        [0, 1],
        [0, 2],
        [1, 0],
        [1, 1],
        [1, 2],
        [2, 0],
        [2, 1],
        [2, 2],
        [0, 0],
        [0, 1],
        [0, 2],
        [1, 0],
        [1, 1],
        [1, 2],
        [2, 0],
        [2, 1],
    ]
    i = 0
    while not memory.is_full:
        episode = EpisodeBuilder()
        obs = env.reset()
        actions = np.array(actions_to_perform[i])
        obs_, reward, done, truncated, info = env.step(actions)
        episode.add(Transition(obs, actions, reward, done, info, obs_, truncated))
        assert episode.is_finished and done
        memory.add(episode.build())
        i += 1

    with open("memory.pkl", "wb") as f:
        pickle.dump(memory, f)
    return memory


test_qplex()
# test_qatten()
# test_qmix()
