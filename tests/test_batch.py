import numpy as np
import torch
from marlenv import Transition
from marlenv.catalog import DiscreteMockEnv

import marl


def _make_batch(size: int, step_reward: float = 1.0, ep_length: int | None = None):
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
    """The constructor's tensors must match the values, dtypes and
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
        "next_available_actions": torch.from_numpy(np.array([t.next_obs.available_actions for t in transitions], dtype=bool)),
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


def test_transition_batch_tensors_are_snapshots_at_construction():
    batch = _make_batch(4)
    batch.transitions[0].reward = np.array([9.0], dtype=np.float32)
    assert batch.rewards[0] == 1.0


def test_transition_minibatch_preserves_modified_tensors_and_metadata(monkeypatch):
    batch = _make_batch(4)
    batch.gamma = torch.tensor([0.8, 0.9, 0.95, 0.99])
    batch.for_individual_learners()
    batch.rewards = batch.rewards + 10
    batch.states = batch.states + 20
    batch.importance_sampling_weights = torch.arange(4, dtype=torch.float32)
    batch._cache["custom"] = torch.arange(4, dtype=torch.float32)

    def unexpected_constructor(*args, **kwargs):
        raise AssertionError("Minibatching must reuse the parent's tensors")

    monkeypatch.setattr(marl.models.batch.TransitionBatch, "__init__", unexpected_constructor)
    indices = [3, 1, 1]
    child = batch.get_minibatch(indices)
    assert child.gamma is batch.gamma
    assert child.device == batch.device
    assert child.reward_size == 1
    torch.testing.assert_close(child.rewards, batch.rewards[indices])
    torch.testing.assert_close(child.states, batch.states[indices])
    torch.testing.assert_close(child.importance_sampling_weights, batch.importance_sampling_weights[indices])
    torch.testing.assert_close(child["custom"], batch["custom"][indices])
    torch.testing.assert_close(child.masks, batch.masks[indices])


EPISODE_FIELDS = (
    "obs",
    "next_obs",
    "all_obs",
    "extras",
    "next_extras",
    "all_extras",
    "states",
    "next_states",
    "all_states",
    "states_extras",
    "next_states_extras",
    "available_actions",
    "next_available_actions",
    "all_available_actions",
    "actions",
    "rewards",
    "dones",
    "masks",
)


def _episodes(time_limits=(3, 7, 5), end_game: int = 6):
    """Episodes of different lengths, some of them done (end_game) and some truncated (time limit). @ai-generated"""
    from marlenv import Builder, Episode

    episodes = []
    for time_limit in time_limits:
        env = Builder(DiscreteMockEnv(end_game=end_game, reward_step=1.5)).time_limit(time_limit, add_extra=False).build()
        obs, state = env.reset()
        episode = Episode.new(obs, state)
        done = False
        while not done:
            action = env.sample_action()
            step = env.step(action)
            transition = Transition.from_step(obs, state, action, step)
            transition["custom"] = np.array([float(len(episode))], dtype=np.float32)
            episode.add(transition)
            obs, state = step.obs, step.state
            done = step.done or step.truncated
        episodes.append(episode)
    return episodes


def _reference_field(episodes, field: str, n_steps: int):
    """How `EpisodeBatch` used to compute each field: from episodes padded with `Episode.padded`. @ai-generated"""
    padded = [e.padded(n_steps) if len(e.actions) < n_steps else e for e in episodes]
    match field:
        case "all_obs":
            values = [e.all_observations for e in padded]
        case "masks":
            return torch.from_numpy(np.array([e.mask for e in padded], dtype=np.float32)).squeeze(-1).transpose(0, 1)
        case "dones":
            return torch.from_numpy(np.array([e.dones for e in padded], dtype=bool).squeeze(-1)).transpose(1, 0)
        case "rewards":
            return torch.from_numpy(np.array([e.rewards for e in padded], dtype=np.float32)).transpose(1, 0).squeeze(-1)
        case "actions":
            return torch.from_numpy(np.array([e.actions for e in padded])).transpose(1, 0)
        case "all_states":
            values = [e.all_states for e in padded]
        case "all_extras" | "states_extras" | "next_states_extras" | "states" | "next_states":
            values = [getattr(e, field) for e in padded]
        case "all_available_actions":
            return torch.from_numpy(np.array([e.all_available_actions for e in padded], dtype=bool)).transpose(1, 0)
        case "available_actions" | "next_available_actions":
            return torch.from_numpy(np.array([getattr(e, field) for e in padded], dtype=bool)).transpose(1, 0)
        case _:
            values = [getattr(e, field) for e in padded]
    return torch.from_numpy(np.array(values, dtype=np.float32)).transpose(1, 0)


def test_episode_batch_fields_match_padded_episodes():
    """Every field must keep the values (including padding), dtypes and shapes of padded episodes. @ai-generated"""
    episodes = _episodes()
    assert any(e.is_done for e in episodes) and any(not e.is_done for e in episodes)
    batch = marl.models.batch.EpisodeBatch(episodes)
    n_steps = max(len(e) for e in episodes)
    for field in EPISODE_FIELDS:
        expected = _reference_field(episodes, field, n_steps)
        actual = getattr(batch, field)
        assert actual.dtype == expected.dtype, f"Mismatch dtype for field {field!r}"
        assert actual.shape == expected.shape, f"Mismatch shape for field {field!r}"
        assert torch.equal(actual, expected), f"Mismatch values for field {field!r}"
    expected_custom = torch.from_numpy(np.array([e["custom"] for e in [e.padded(n_steps) for e in episodes]], dtype=np.float32))
    assert torch.equal(batch["custom"], expected_custom.transpose(1, 0))


def test_episode_minibatch_keeps_the_time_dimension_of_its_parent():
    """PPO indexes parent tensors with minibatch tensors, so both need the same time dimension. @ai-generated"""
    episodes = _episodes()
    batch = marl.models.batch.EpisodeBatch(episodes)
    minibatch = batch.get_minibatch([0, 2])  # The two shortest episodes
    n_steps = max(len(e) for e in episodes)
    for field in EPISODE_FIELDS:
        expected = getattr(batch, field)[:, [0, 2]]
        actual = getattr(minibatch, field)
        assert actual.shape == expected.shape, f"Mismatch shape for field {field!r}"
        assert torch.equal(actual, expected), f"Mismatch values for field {field!r}"
        assert actual.shape[0] in (n_steps, n_steps + 1)


def test_episode_batch_packed_observations_match_all_obs():
    """Packing and unpacking must round-trip the non-padding items, whichever is computed first. @ai-generated"""
    episodes = _episodes()
    for all_obs_first in (True, False):
        batch = marl.models.batch.EpisodeBatch(episodes)
        if all_obs_first:
            batch.all_obs
        packed = batch.packed_all_obs
        assert packed.shape[0] == sum(len(e) + 1 for e in episodes)
        unpacked = batch.unpack_all(packed)
        valid = batch.all_masks.bool()
        assert torch.equal(unpacked[valid], batch.all_obs[valid])
        assert torch.all(unpacked[~valid] == 0)
        assert torch.equal(batch.unpack_all(batch.packed_all_extras), batch.all_extras)


def test_transition_batch_extend_preserves_metadata():
    batch = _make_batch(4)
    batch.gamma = torch.tensor(0.95)
    extended = batch.extend(batch.transitions[:2])
    assert extended.gamma is batch.gamma
    assert extended.device == batch.device
    assert extended.size == 6
    torch.testing.assert_close(extended.rewards, torch.ones(6))
