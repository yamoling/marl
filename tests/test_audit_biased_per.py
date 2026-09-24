"""Regression tests for unsupported biased/prioritized replay compositions."""

import pytest
from marlenv import Episode, Transition
from marlenv.catalog import DiscreteMockEnv

from marl.models.replay_memory import BiasedMemory, PrioritizedMemory, TransitionMemory


def _episode() -> Episode:
    """Build a single-transition demonstration. @ai-generated"""
    env = DiscreteMockEnv(end_game=1)
    obs, state = env.reset()
    action = env.sample_action()
    episode = Episode.new(obs, state)
    episode.add(Transition.from_step(obs, state, action, env.step(action)))
    return episode


def test_biased_memory_rejects_prioritized_memory():
    """A biased sampler must not silently bypass PER. @ai-generated"""
    with pytest.raises(TypeError, match="BiasedMemory cannot wrap PrioritizedMemory.*bypasses priorities"):
        BiasedMemory.from_episodes([_episode()], PrioritizedMemory(TransitionMemory(10)))


def test_prioritized_memory_rejects_biased_memory():
    """Tree slots must not silently address demonstrations instead of agent items. @ai-generated"""
    biased = BiasedMemory.from_episodes([_episode()], TransitionMemory(10))
    with pytest.raises(TypeError, match="PrioritizedMemory cannot wrap BiasedMemory.*priority tree slots"):
        PrioritizedMemory(biased)
