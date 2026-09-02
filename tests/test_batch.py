from typing import Optional

import numpy as np
import torch
from marlenv import Transition
from marlenv.catalog import DiscreteMockEnv

import marl


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
        transitions.append(Transition.from_step(obs, state, action, step))
    return marl.models.batch.TransitionBatch(transitions)


def test_transition_batch_creation():
    batch = _make_batch(10, step_reward=1.0)
    assert len(batch) == 10
    assert batch.size == 10
    assert batch.dones[-1]
    assert torch.all(~batch.dones[:-2])
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
    actual = batch.compute_mc_returns(GAMMA, 0.0)
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
    actual = batch.compute_mc_returns(GAMMA, 0.0)
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
    values = all_values[:-1]
    next_values = all_values[1:]

    gae_0 = batch.compute_gae(GAMMA, values, next_values, trace_decay=0, normalize=False)
    td1 = batch.compute_td1_advantages(GAMMA, all_values, normalize=False)

    assert torch.allclose(gae_0, td1)


def test_gae1_is_mc():
    REWARD_STEP = 1.5
    GAMMA = 0.99
    EP_LENGTH = 10
    batch = _make_batch(EP_LENGTH, step_reward=REWARD_STEP)

    all_values = torch.rand(EP_LENGTH + 1, dtype=torch.float32)
    values = all_values[:-1]
    next_values = all_values[1:]

    gae_1 = batch.compute_gae(GAMMA, values, next_values, trace_decay=1.0, normalize=False)
    mc = batch.compute_mc_advantages(GAMMA, all_values, normalize=False)

    assert torch.allclose(gae_1, mc)


def test_transition_batch_get_minibatch_matches_fresh_batch():
    """The device-indexing fast path of `get_minibatch` must produce the same tensors as building
    a fresh `TransitionBatch` from the corresponding subset of transitions."""
    batch = _make_batch(20, step_reward=1.5)
    indices = [1, 3, 4, 7, 12, 19]

    # Force materialization of every field on the parent batch, as `PPO.train` does before entering
    # the epoch loop.
    for field in (
        "obs",
        "next_obs",
        "extras",
        "next_extras",
        "actions",
        "rewards",
        "dones",
        "available_actions",
        "masks",
    ):
        getattr(batch, field)

    minibatch = batch.get_minibatch(indices)
    expected = marl.models.batch.TransitionBatch([batch.transitions[i] for i in indices])

    assert minibatch.size == expected.size
    for field in (
        "obs",
        "next_obs",
        "extras",
        "next_extras",
        "actions",
        "rewards",
        "dones",
        "available_actions",
        "masks",
    ):
        actual_value = getattr(minibatch, field)
        expected_value = getattr(expected, field)
        assert torch.equal(actual_value, expected_value), f"Mismatch for field {field!r}"


def test_transition_batch_get_minibatch_unmaterialized_field_still_lazy():
    """Fields never accessed on the parent batch must still be computable (lazily) on the minibatch."""
    batch = _make_batch(20, step_reward=1.5)
    indices = [0, 5, 10]

    minibatch = batch.get_minibatch(indices)
    expected = marl.models.batch.TransitionBatch([batch.transitions[i] for i in indices])

    assert torch.equal(minibatch.states, expected.states)
    assert torch.equal(minibatch.next_states, expected.next_states)


def test_transition_batch_for_individual_learners_order_independent_of_minibatching():
    """Applying `for_individual_learners` before or after `get_minibatch` must give the same result,
    and applying it twice on the resulting minibatch (as `PPO.train` does) must be a no-op."""
    indices = [2, 6, 9, 15]

    # Order 1: expand on the parent, then slice. Simulate PPO calling `for_individual_learners` again on
    # the resulting minibatch: it must be a no-op since the tensors are already agent-wise.
    parent_first = _make_batch(20, step_reward=1.5)
    parent_first.for_individual_learners()
    minibatch_from_parent = parent_first.get_minibatch(indices)
    minibatch_from_parent.for_individual_learners()

    # Order 2: slice first (from an equivalent, not-yet-expanded batch), then expand on the child only.
    batch = _make_batch(20, step_reward=1.5)
    minibatch_then_expanded = batch.get_minibatch(indices)
    minibatch_then_expanded.for_individual_learners()

    assert torch.equal(minibatch_from_parent.rewards, minibatch_then_expanded.rewards)
    assert torch.equal(minibatch_from_parent.dones, minibatch_then_expanded.dones)
    assert torch.equal(minibatch_from_parent.masks, minibatch_then_expanded.masks)


def test_transition_batch_single_pass_packing_matches_reference():
    """The single-pass-packed fields (`TransitionBatch._pack`) must match the values, dtypes and
    shapes of the reference per-field computation (`np.array([t.<field> for t in transitions])` then
    `torch.from_numpy`), which is how each field used to be computed independently.

    @ai-generated
    """
    batch = _make_batch(16, step_reward=1.5)
    transitions = batch.transitions

    fresh = marl.models.batch.TransitionBatch(transitions)

    reference = {
        "obs": torch.from_numpy(np.array([t.obs.data for t in transitions], dtype=np.float32)),
        "next_obs": torch.from_numpy(np.array([t.next_obs.data for t in transitions], dtype=np.float32)),
        "extras": torch.from_numpy(np.array([t.obs.extras for t in transitions], dtype=np.float32)),
        "next_extras": torch.from_numpy(np.array([t.next_obs.extras for t in transitions], dtype=np.float32)),
        "actions": torch.from_numpy(np.array([t.action for t in transitions])),
        "rewards": torch.from_numpy(np.array([t.reward for t in transitions], dtype=np.float32)).squeeze(-1),
        "available_actions": torch.from_numpy(np.array([t.obs.available_actions for t in transitions], dtype=bool)),
        "next_available_actions": torch.from_numpy(
            np.array([t.next_obs.available_actions for t in transitions], dtype=bool)
        ),
    }
    np_dones = np.array([t.done for t in transitions], dtype=bool)
    dones = torch.from_numpy(np_dones)
    if reference["rewards"].dim() > 1:
        dones = dones.unsqueeze(-1).expand_as(reference["rewards"])
    reference["dones"] = dones

    for field, expected in reference.items():
        actual = getattr(fresh, field)
        assert actual.dtype == expected.dtype, f"Mismatch dtype for field {field!r}"
        assert actual.shape == expected.shape, f"Mismatch shape for field {field!r}"
        assert torch.equal(actual, expected), f"Mismatch values for field {field!r}"

    # `masks` is allocated directly on the batch's device rather than moved after the fact.
    assert torch.equal(fresh.masks, torch.ones(len(transitions)))
