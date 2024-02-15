import torch
from marl.models.batch import EpisodeBatch
from rlenv.wrappers import TimeLimit
from .utils import generate_episode, MockEnv


def test_episode_batch_padded():
    env = TimeLimit(MockEnv(2), 5)
    episodes = []
    for i in range(5, 10):
        env.step_limit = i
        episodes.append(generate_episode(env))
    batch = EpisodeBatch(episodes)
    assert len(batch.obs) == 9
    assert len(batch.obs_) == 10
    assert len(batch.masks) == 9


def test_episode_batch_returns():
    episodes = []
    GAMMA = 0.9
    for i in range(5):
        MockEnv.END_GAME = i + 1  # type: ignore
        env = MockEnv(2)
        episodes.append(generate_episode(env))

    batch = EpisodeBatch(episodes)
    returns = batch.compute_returns(GAMMA)
    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.9, 1.0, 0.0, 0.0, 0.0],
            [2.71, 1.9, 1.0, 0.0, 0.0],
            [3.439, 2.71, 1.9, 1.0, 0.0],
            [4.0951, 3.439, 2.71, 1.9, 1.0],
        ]
    ).transpose(1, 0)
    assert torch.allclose(returns, expected)
