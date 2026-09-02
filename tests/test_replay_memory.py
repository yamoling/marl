"""
Tests for marl.models.replay_memory: TransitionMemory, EpisodeMemory, NStepMemory,
BiasedMemory and PrioritizedMemory.
"""

import numpy as np
import pytest
import torch
from marlenv import Episode, Transition
from marlenv.catalog import DiscreteMockEnv

from marl.models.batch import EpisodeBatch, TransitionBatch
from marl.models.replay_memory import EpisodeMemory, TransitionMemory
from marl.models.replay_memory.biased_memory import BiasedMemory
from marl.models.replay_memory.nstep_memory import NStepMemory
from marl.models.replay_memory.prioritized_memory import PrioritizedMemory


def _make_transitions(n: int, end_game: int = 1000, reward_step: float = 1.0):
    env = DiscreteMockEnv(end_game=end_game, reward_step=reward_step)
    obs, state = env.reset()
    transitions = []
    for _ in range(n):
        action = env.sample_action()
        step = env.step(action)
        transitions.append(Transition.from_step(obs, state, action, step))
        obs, state = step.obs, step.state
        if step.done:
            obs, state = env.reset()
    return transitions


def _make_episode(length: int) -> Episode:
    env = DiscreteMockEnv(end_game=length)
    obs, state = env.reset()
    episode = Episode.new(obs, state)
    for _ in range(length):
        action = env.sample_action()
        step = env.step(action)
        episode.add(Transition.from_step(obs, state, action, step))
        obs, state = step.obs, step.state
    return episode


class TestTransitionMemory:
    def test_starts_empty(self):
        assert len(TransitionMemory(10)) == 0

    def test_update_on_is_transition(self):
        memory = TransitionMemory(10)
        assert memory.update_on == "transition"
        assert memory.update_on_transitions
        assert not memory.update_on_episodes

    def test_add_transition_increases_length(self):
        memory = TransitionMemory(10)
        for t in _make_transitions(3):
            memory.add_transition(t)
        assert len(memory) == 3

    def test_generic_add_also_works(self):
        memory = TransitionMemory(10)
        memory.add(_make_transitions(1)[0])
        assert len(memory) == 1

    def test_respects_max_size(self):
        memory = TransitionMemory(5)
        for t in _make_transitions(20):
            memory.add(t)
        assert len(memory) == 5
        assert memory.is_full

    def test_can_sample_returns_false_when_too_small(self):
        memory = TransitionMemory(10)
        memory.add(_make_transitions(1)[0])
        assert not memory.can_sample(5)
        assert memory.can_sample(1)

    def test_sample_returns_a_transition_batch_of_requested_size(self):
        memory = TransitionMemory(10)
        for t in _make_transitions(10):
            memory.add(t)
        batch = memory.sample(4)
        assert isinstance(batch, TransitionBatch)
        assert batch.size == 4

    def test_as_batch_contains_every_item(self):
        memory = TransitionMemory(10)
        transitions = _make_transitions(6)
        for t in transitions:
            memory.add(t)
        batch = memory.as_batch()
        assert batch.size == 6

    def test_clear_empties_the_memory(self):
        memory = TransitionMemory(10)
        for t in _make_transitions(4):
            memory.add(t)
        memory.clear()
        assert len(memory) == 0

    def test_getitem_returns_the_stored_transition(self):
        memory = TransitionMemory(10)
        transitions = _make_transitions(3)
        for t in transitions:
            memory.add(t)
        assert memory[0] is transitions[0]

    def test_update_is_a_no_op_returning_empty_logs(self):
        memory = TransitionMemory(10)
        assert memory.update(0) == {}


class TestEpisodeMemory:
    def test_update_on_is_episode(self):
        memory = EpisodeMemory(10)
        assert memory.update_on == "episode"
        assert memory.update_on_episodes
        assert not memory.update_on_transitions

    def test_add_episode_increases_length(self):
        memory = EpisodeMemory(10)
        memory.add_episode(_make_episode(5))
        assert len(memory) == 1

    def test_sample_returns_episode_batch(self):
        memory = EpisodeMemory(10)
        for _ in range(5):
            memory.add(_make_episode(4))
        batch = memory.sample(3)
        assert isinstance(batch, EpisodeBatch)


class TestNStepMemory:
    def test_length_is_reduced_by_n_before_episode_end(self):
        memory = NStepMemory(100, n=3, gamma=0.9)
        for t in _make_transitions(5, end_game=1000):
            memory.add(t)
        # 5 transitions added, none terminal: the last n=3 are held back.
        assert len(memory) == 5 - 3

    def test_n_step_return_matches_manual_computation(self):
        gamma = 0.9
        n = 3
        memory = NStepMemory(100, n=n, gamma=gamma)
        transitions = _make_transitions(n, end_game=1000, reward_step=1.0)
        for t in transitions:
            memory.add(t)
        # Not yet terminal, so nothing should have been finalised: len == 0.
        assert len(memory) == 0
        # Manually recompute the n-step discounted reward for the first transition.
        expected = sum(gamma**i * transitions[i].reward.item() for i in range(n))
        stored = memory[0]
        assert stored.reward.item() == pytest.approx(expected)

    def test_terminal_transition_updates_next_obs_of_previous_steps(self):
        n = 3
        memory = NStepMemory(100, n=n, gamma=0.9)
        transitions = _make_transitions(n, end_game=n, reward_step=1.0)
        for t in transitions:
            memory.add(t)
        assert transitions[-1].done
        last = memory[-1]
        assert last.done
        # The transition n-2 steps before the terminal one should now point to the terminal next_obs.
        earlier = memory[-2]
        assert np.array_equal(earlier.next_obs.data, transitions[-1].next_obs.data)
        assert earlier.done == transitions[-1].done


class TestBiasedMemory:
    def test_length_includes_bias_and_wrapped_memory(self):
        bias = _make_transitions(3)
        memory = BiasedMemory.from_transitions(bias, max_size=20)
        assert len(memory) == 3
        memory.add(_make_transitions(1)[0])
        assert len(memory) == 4

    def test_bias_items_come_first(self):
        bias = _make_transitions(2)
        memory = BiasedMemory.from_transitions(bias, max_size=20)
        assert memory[0] is bias[0]
        assert memory[1] is bias[1]

    def test_raises_if_bias_is_empty(self):
        with pytest.raises(AssertionError):
            BiasedMemory.from_transitions([], max_size=20)

    def test_raises_for_non_positive_factor(self):
        with pytest.raises(AssertionError):
            BiasedMemory.from_transitions(_make_transitions(1), max_size=20, factor=0.0)

    def test_sample_returns_requested_batch_size(self):
        bias = _make_transitions(2)
        memory = BiasedMemory.from_transitions(bias, max_size=20)
        for t in _make_transitions(5):
            memory.add(t)
        batch = memory.sample(4)
        assert batch.size == 4

    def test_high_factor_strongly_prefers_biased_items(self):
        np.random.seed(0)
        bias = _make_transitions(1, reward_step=42.0)
        memory = BiasedMemory.from_transitions(bias, max_size=1000, factor=1e6)
        for t in _make_transitions(50, reward_step=1.0):
            memory.add(t)
        # With an overwhelming bias factor, batch_size=1 samples should almost always hit the biased item.
        hits = sum(1 for _ in range(20) if memory.sample(1).rewards.item() == pytest.approx(42.0))
        assert hits >= 15

    def test_clear_delegates_to_wrapped_memory(self):
        bias = _make_transitions(1)
        memory = BiasedMemory.from_transitions(bias, max_size=20)
        for t in _make_transitions(3):
            memory.add(t)
        memory.clear()
        assert len(memory) == 1  # bias items remain


class TestPrioritizedMemory:
    def test_wraps_add_and_length(self):
        inner = TransitionMemory(10)
        memory = PrioritizedMemory(inner, multi_objective=False)
        for t in _make_transitions(4):
            memory.add(t)
        assert len(memory) == 4

    def test_alpha_and_beta_accept_floats(self):
        inner = TransitionMemory(10)
        memory = PrioritizedMemory(inner, multi_objective=False, alpha=0.6, beta=0.5)
        assert memory.alpha.value == pytest.approx(0.6)
        assert memory.beta.value == pytest.approx(0.5)

    def test_sample_produces_normalised_importance_weights(self):
        np.random.seed(0)
        inner = TransitionMemory(10)
        memory = PrioritizedMemory(inner, multi_objective=False, beta=0.5)
        for t in _make_transitions(8):
            memory.add(t)
        batch = memory.sample(4)
        weights = batch.importance_sampling_weights
        assert weights is not None
        assert torch.max(weights).item() == pytest.approx(1.0)
        assert torch.all(weights > 0)

    def test_update_raises_without_td_error(self):
        inner = TransitionMemory(10)
        memory = PrioritizedMemory(inner, multi_objective=False)
        for t in _make_transitions(4):
            memory.add(t)
        memory.sample(2)
        with pytest.raises(ValueError):
            memory.update(0)

    def test_update_increases_priority_for_high_td_error(self):
        inner = TransitionMemory(10)
        memory = PrioritizedMemory(inner, multi_objective=False, alpha=1.0, eps=1e-2, td_error_clipping=None)
        for t in _make_transitions(5):
            memory.add(t)
        memory.sample(3)
        logs = memory.update(0, td_error=torch.tensor([10.0, 0.0, 0.0]))
        assert logs["mean-priority"] > 0
        assert memory.max_priority >= 10.0

    def test_invalid_alpha_type_raises(self):
        inner = TransitionMemory(10)
        with pytest.raises(ValueError):
            PrioritizedMemory(inner, multi_objective=False, alpha="bad")  # type: ignore[arg-type]
