from typing import Optional
import marl
import torch

from .two_steps import TwoSteps, State


class QNetwork(marl.nn.LinearNN):
    def __init__(self, input_shape: tuple[int, ...], extras_shape: Optional[tuple[int, ...]], output_shape: tuple[int, ...]):
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
    mixer = marl.qlearning.QMix(env.state_shape[0], env.n_agents, embed_size=8)
    # mixer = marl.qlearning.VDN(2)
    trainer = marl.training.DQNTrainer(
        qnetwork=qnetwork,
        train_policy=policy,
        double_qlearning=True,
        memory=memory,
        batch_size=32,
        train_every="episode",
        update_interval=1,
        mixer=mixer,
        target_updater=marl.training.HardUpdate(100),
        gamma=0.99,
        optimiser="rmsprop",
        lr=5e-4,
    )
    trainer.show()
    algo = marl.qlearning.DQN(qnetwork, policy)
    exp = marl.Experiment.create("logs/test", algo, trainer, env, 10_000, 10_000)
    runner = exp.create_runner(0)
    runner.train(0)

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
                qs = torch.tensor([qvalues[0][a0], qvalues[1][a1]]).unsqueeze(0)
                s = torch.tensor(obs.state, dtype=torch.float32).unsqueeze(0)
                res = mixer.forward(qs, s).detach()
                payoff_matrix[a0][a1] = res.item()
        assert np.allclose(np.array(payoff_matrix), np.array(expected[state]), atol=0.1)
