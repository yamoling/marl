import numpy as np
import torch
import random
from copy import deepcopy
from marl.models.batch import EpisodeBatch, TransitionBatch, Batch
from marlenv.wrappers import TimeLimit
from marlenv import EpisodeBuilder, Transition, Builder, MockEnv
from .utils import generate_episode


def test_episode_batch_padded():
    env = TimeLimit(MockEnv(2), 5)
    episodes = []
    for i in range(5, 10):
        env.step_limit = i
        episodes.append(generate_episode(env))
    batch = EpisodeBatch(episodes)
    assert len(batch.obs) == 9
    assert len(batch.obs_) == 9
    assert len(batch.masks) == 9


def test_episode_batch_returns():
    episodes = []
    GAMMA = 0.9
    for i in range(5):
        env = MockEnv(2, end_game=i + 1)
        episodes.append(generate_episode(env))

    batch = EpisodeBatch(episodes)
    returns = batch.compute_returns(GAMMA)
    expected = (
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [1.9, 1.0, 0.0, 0.0, 0.0],
                [2.71, 1.9, 1.0, 0.0, 0.0],
                [3.439, 2.71, 1.9, 1.0, 0.0],
                [4.0951, 3.439, 2.71, 1.9, 1.0],
            ]
        )
        .transpose(1, 0)
        .unsqueeze(-1)  # Unsqueeze the reward dimension
    )
    assert torch.allclose(returns, expected)


def _batch_test(batch: Batch):
    assert (
        len(batch.obs)
        == len(batch.obs_)
        == len(batch.extras)
        == len(batch.extras_)
        == len(batch.states)
        == len(batch.states_)
        == len(batch.available_actions)
        == len(batch.available_actions_)
        == len(batch.masks)
        == len(batch.rewards)
    )
    assert len(batch.all_obs_) == len(batch.all_extras_)
    assert torch.equal(batch.all_obs_[0], batch.obs[0])
    for i in range(len(batch)):
        assert torch.equal(batch.obs_[i], batch.all_obs_[i + 1])
        assert torch.equal(batch.extras_[i], batch.all_extras_[i + 1])


def test_transition_batch():
    env = Builder(MockEnv(2)).time_limit(10).build()
    episode = generate_episode(env)
    transitions = list(episode.transitions())
    random.shuffle(transitions)
    batch = TransitionBatch(transitions)

    obs = torch.from_numpy(np.array([t.obs.data for t in transitions]))
    obs_ = torch.from_numpy(np.array([t.obs_.data for t in transitions]))
    assert torch.equal(batch.obs, obs)
    assert torch.equal(batch.obs_, obs_)
    assert torch.equal(batch.all_obs_[1:], obs_)
    assert torch.equal(batch.all_obs_[0], obs[0])
    _batch_test(batch)


def test_episode_batch():
    env = Builder(MockEnv(2)).time_limit(10).build()
    episode = generate_episode(env)
    episode2 = deepcopy(episode)
    episode2._observations = np.random.random(episode2._observations.shape).astype(np.float32)
    episode3 = deepcopy(episode)
    episode3._observations = np.random.random(episode3._observations.shape).astype(np.float32)
    batch = EpisodeBatch([episode, episode2, episode3])
    _batch_test(batch)
