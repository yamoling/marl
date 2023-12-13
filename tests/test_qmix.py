from typing import Optional
import marl
import torch

from .two_steps import TwoSteps


class QNetwork(marl.nn.LinearNN):
    def __init__(self, input_shape: tuple[int, ...], extras_shape: Optional[tuple[int, ...]], output_shape: tuple[int, ...]):
        super().__init__(input_shape, extras_shape, output_shape)
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(input_shape[0], 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_shape[0]),
        )

    def forward(self, obs, extras):
        return self.nn(obs)


def test_qmix_value():
    """
    Demonstration of QMix higher representation capabilities as described in the paper.

    https://arxiv.org/pdf/1803.11485.pdf
    Appendix B.
    """
    env = TwoSteps()
    env.reset()

    qnetwork = QNetwork.from_env(env)
    policy = marl.policy.EpsilonGreedy.constant(1.0)
    memory = marl.models.EpisodeMemory(500)
    trainer = marl.training.DQNTrainer(
        qnetwork=qnetwork,
        train_policy=policy,
        memory=memory,
        batch_size=32,
        train_every="episode",
        mixer=marl.qlearning.QMix(
            env.state_shape[0],
            env.n_agents,
        ),
        gamma=0.99,
        optimiser="rmsprop",
        lr=5e-4,
    )
    algo = marl.qlearning.DQN(qnetwork, policy)

    runner = marl.models.Runner(env, algo, trainer, "logs/tests", 5000, 10_000)
    runner.train(0)
