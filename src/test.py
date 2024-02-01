import marl
import torch

from enum import IntEnum
from typing import Literal
import rlenv
import numpy as np
import numpy.typing as npt
from rlenv.models import DiscreteActionSpace, Observation

marl.seed(0)

PAYOFF_INITIAL = [[0, 0], [0, 0]]

PAYOFF_2A = [[7, 7], [7, 7]]

PAYOFF_2B = [[0, 1], [1, 8]]


class State(IntEnum):
    INITIAL = 0
    STATE_2A = 1
    STATE_2B = 2
    END = 3

    def one_hot(self):
        res = np.zeros((4,), dtype=np.float32)
        res[self.value] = 1
        return res


class TwoSteps(rlenv.RLEnv[DiscreteActionSpace]):
    def __init__(self):
        self.state = State.INITIAL
        self._identity = np.identity(2, dtype=np.float32)
        super().__init__(
            DiscreteActionSpace(2, 2),
            observation_shape=(self.state.one_hot().shape[0] + 2,),
            state_shape=self.state.one_hot().shape,
        )

    def reset(self) -> Observation:
        self.state = State.INITIAL
        return self.observation()

    def step(self, actions: npt.NDArray[np.int32]):
        match self.state:
            case State.INITIAL:
                # In the initial step, only agent 0's actions have an influence on the state
                payoffs = PAYOFF_INITIAL
                if actions[0] == 0:
                    self.state = State.STATE_2A
                elif actions[0] == 1:
                    self.state = State.STATE_2B
                else:
                    raise ValueError(f"Invalid action: {actions[0]}")
            case State.STATE_2A:
                payoffs = PAYOFF_2A
                self.state = State.END
            case State.STATE_2B:
                payoffs = PAYOFF_2B
                self.state = State.END
            case State.END:
                raise ValueError("Episode is already over")
        obs = self.observation()
        reward = payoffs[actions[0]][actions[1]]
        done = self.state == State.END
        return obs, reward, done, False, {}

    def get_state(self):
        return self.state.one_hot()

    def observation(self):
        obs_data = np.array([self.state.one_hot(), self.state.one_hot()])
        extras = self._identity
        return Observation(obs_data, self.available_actions(), self.get_state(), extras)

    def render(self, mode: Literal["human", "rgb_array"] = "human"):
        print(self.state)

    def force_state(self, state: State):
        self.state = state

    def seed(self, seed):
        return


class QNetwork(marl.nn.LinearNN):
    def __init__(self, input_shape: tuple[int, ...], extras_shape: tuple[int, ...], output_shape: tuple[int, ...]):
        super().__init__(input_shape, extras_shape, output_shape)
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(input_shape[0], 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_shape[0]),
        )

    def forward(self, obs, extras):
        if extras is not None:
            obs = torch.concatenate([obs, extras], dim=-1)
        return self.nn(obs)


def test_qmix_value(device):
    """
    Demonstration of QMix higher representation capabilities against VDN as described in the paper.
    Appendix B.1.

    https://arxiv.org/pdf/1803.11485.pdf
    Appendix B.
    """
    env = TwoSteps()
    env.reset()

    qnetwork = QNetwork.from_env(env)
    policy = marl.policy.EpsilonGreedy.constant(1.0)
    memory = marl.models.EpisodeMemory(500)
    mixer = marl.qlearning.QMix(env.state_shape[0], env.n_agents, embed_size=8)
    # mixer = marl.qlearning.VDN(2)
    device = marl.utils.get_device(device)
    trainer = marl.training.DQNTrainer(
        qnetwork=qnetwork,
        train_policy=policy,
        double_qlearning=True,
        memory=memory,
        batch_size=32,
        train_interval=(1, "episode"),
        mixer=mixer,
        target_updater=marl.training.HardUpdate(100),
        gamma=0.99,
        optimiser="rmsprop",
        lr=5e-4,
    )
    # trainer.show()
    algo = marl.qlearning.DQN(qnetwork, policy)
    exp = marl.Experiment.create("logs/test", algo, trainer, env, 10_000, 10_000)
    runner = exp.create_runner(0)
    runner.to(device)
    runner.train(0)

    # Expected results shown in the paper
    expected = {
        State.INITIAL: [[6.93, 6.93], [7.92, 7.92]],
        State.STATE_2A: [[7.0, 7.0], [7.0, 7.0]],
        State.STATE_2B: [[0, 1], [1, 8]],
    }

    for state in State.INITIAL, State.STATE_2A, State.STATE_2B:
        env.force_state(state)
        obs = env.observation()
        qvalues = algo.compute_qvalues(obs)
        import numpy as np

        payoff_matrix = [[0, 0], [0, 0]]
        for a0 in range(2):
            for a1 in range(2):
                qs = torch.tensor([qvalues[0][a0], qvalues[1][a1]]).unsqueeze(0).to(device)
                s = torch.tensor(obs.state, dtype=torch.float32).unsqueeze(0).to(device)
                res = mixer.forward(qs, s).detach()
                payoff_matrix[a0][a1] = res.item()
        print(payoff_matrix)
        # assert np.allclose(np.array(payoff_matrix), np.array(expected[state]), atol=0.1)


test_qmix_value("cpu")
# test_qmix_value("cuda")
