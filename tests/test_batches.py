import torch
import random
from marl.models.batch import EpisodeBatch, TransitionBatch, Batch
from rlenv.wrappers import TimeLimit
from rlenv import EpisodeBuilder, Transition, Builder, MockEnv
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
        MockEnv.END_GAME = i + 1  # type: ignore
        env = MockEnv(2)
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
        batch.size
        == len(batch.obs)
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
    assert batch.size + 1 == len(batch.all_obs) == len(batch.all_extras) == len(batch.all_states) == len(batch.all_available_actions)
    for i in range(len(batch)):
        assert torch.equal(batch.obs[i], batch.all_obs[i])
        assert torch.equal(batch.obs_[i], batch.all_obs[i + 1])
        assert torch.equal(batch.extras[i], batch.all_extras[i])
        assert torch.equal(batch.extras_[i], batch.all_extras[i + 1])
        assert torch.equal(batch.states[i], batch.all_states[i])
        assert torch.equal(batch.states_[i], batch.all_states[i + 1])
        assert torch.equal(batch.available_actions[i], batch.all_available_actions[i])
        assert torch.equal(batch.available_actions_[i], batch.all_available_actions[i + 1])


def test_transition_batch_obs():
    env = Builder(MockEnv(2)).time_limit(4).build()
    episode = generate_episode(env)
    transitions = list(episode.transitions())
    random.shuffle(transitions)
    batch = TransitionBatch(transitions)
    _batch_test(batch)


def test_episode_batch_obs():
    env = Builder(MockEnv(2)).time_limit(4).build()
    episode = generate_episode(env)
    batch = EpisodeBatch([episode, episode, episode])
    _batch_test(batch)
