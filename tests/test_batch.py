from typing import Optional
from marlenv import Transition, DiscreteMockEnv
import marl
import torch


def _make_batch(size: int, step_reward: float = 1.0, ep_length: Optional[int] = None):
    if ep_length is None:
        ep_length = size
    env = DiscreteMockEnv(end_game=ep_length, reward_step=step_reward)
    obs, state = env.reset()
    transitions = list[Transition]()
    t = 0
    done = False
    while t < size:
        t += 1
        if done:
            obs, state = env.reset()
            done = False
        action = env.sample_action()
        step = env.step(action)
        done = step.done
        transitions.append(Transition.from_step(obs, state, action, step))  # type: ignore
    return marl.models.batch.TransitionBatch(transitions)


def test_transition_batch_creation():
    batch = _make_batch(10, step_reward=1.0)
    assert len(batch) == 10
    assert batch.size == 10
    assert batch.done_masks[-1]
    assert torch.all(~batch.done_masks[:-2])
    assert torch.all(batch.rewards == 1.0)


def test_batch_mc_returns():
    REWARD_STEP = 1.5
    GAMMA = 0.99
    EP_LENGTH = 10
    batch = _make_batch(EP_LENGTH, step_reward=REWARD_STEP)

    expected_returns = []
    for i in range(EP_LENGTH):
        g = 0.0
        for j in range(i, EP_LENGTH):
            g += GAMMA ** (j - i) * REWARD_STEP
        expected_returns.append(g)
    expected_returns = torch.tensor(expected_returns, dtype=torch.float32)
    actual = batch.compute_mc_returns(GAMMA, torch.zeros(1), normalize=False)
    assert torch.allclose(actual, expected_returns)


def test_batch_mc_returns_episode_ended():
    REWARD_STEP = 1.5
    GAMMA = 0.99
    EP_LENGTH = 5
    BATCH_SIZE = 10
    batch = _make_batch(BATCH_SIZE, step_reward=REWARD_STEP, ep_length=EP_LENGTH)

    expected_returns = []
    for i in range(EP_LENGTH):
        g = 0.0
        for j in range(i, EP_LENGTH):
            g += GAMMA ** (j - i) * REWARD_STEP
        expected_returns.append(g)
    for i in range(EP_LENGTH):
        g = 0.0
        for j in range(i, EP_LENGTH):
            g += GAMMA ** (j - i) * REWARD_STEP
        expected_returns.append(g)
    expected_returns = torch.tensor(expected_returns, dtype=torch.float32)
    actual = batch.compute_mc_returns(GAMMA, torch.zeros(1), normalize=False)
    assert torch.allclose(actual, expected_returns)


def test_batch_td1_returns():
    REWARD_STEP = 1.5
    GAMMA = 0.99
    EP_LENGTH = 10
    batch = _make_batch(EP_LENGTH, step_reward=REWARD_STEP)

    next_values = torch.zeros(EP_LENGTH, dtype=torch.float32)
    expected = torch.full((EP_LENGTH,), REWARD_STEP)
    actual = batch.compute_td1_returns(GAMMA, next_values, normalize=False)
    assert torch.allclose(actual, expected)


def test_batch_td1_returns_episode_ended():
    REWARD_STEP = 1.5
    GAMMA = 0.99
    EP_LENGTH = 10
    batch = _make_batch(EP_LENGTH, step_reward=REWARD_STEP, ep_length=EP_LENGTH // 2)

    next_values = torch.zeros(EP_LENGTH, dtype=torch.float32)
    expected = torch.full((EP_LENGTH,), REWARD_STEP)
    actual = batch.compute_td1_returns(GAMMA, next_values, normalize=False)
    assert torch.allclose(actual, expected)


def test_gae0_is_td1():
    REWARD_STEP = 1.5
    GAMMA = 0.99
    EP_LENGTH = 10
    batch = _make_batch(EP_LENGTH, step_reward=REWARD_STEP)

    all_values = torch.rand(EP_LENGTH + 1, dtype=torch.float32)

    gae_0 = batch.compute_gae(GAMMA, all_values, trace_decay=0, normalize=False)
    td1 = batch.compute_td1_advantages(GAMMA, all_values, normalize=False)

    assert torch.allclose(gae_0, td1)


def test_gae1_is_mc():
    REWARD_STEP = 1.5
    GAMMA = 0.99
    EP_LENGTH = 10
    batch = _make_batch(EP_LENGTH, step_reward=REWARD_STEP)

    all_values = torch.rand(EP_LENGTH + 1, dtype=torch.float32)

    gae_1 = batch.compute_gae(GAMMA, all_values, trace_decay=1.0, normalize=False)
    mc = batch.compute_mc_advantages(GAMMA, all_values, normalize=False)

    assert torch.allclose(gae_1, mc)
