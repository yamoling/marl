from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from marlenv import Episode, Transition

from .replay_memory import EpisodeMemory, ReplayMemory, TransitionMemory


@dataclass
class BiasedMemory[T](ReplayMemory[T]):
    n_bias: int
    wrapped: ReplayMemory[T]
    factor: float
    """Factor that multiplies the probability of sampling biased items."""

    def __init__(self, bias: Iterable[T], memory: ReplayMemory[T], factor: float = 1.0):
        bias = list(bias)
        assert len(bias) < memory.max_size, "The bias should be smaller than the memory size"
        assert len(bias) > 0, "There sould be at least one element to bias towards"
        assert factor > 0, "factor must be greater than 0"
        super().__init__(memory.max_size + len(bias), memory.update_on)
        self._memory.extend(bias)
        self.n_bias = len(bias)
        self.wrapped = memory
        self.factor = factor

    def add(self, item: T):
        return self.wrapped.add(item)

    def add_transition(self, transition: Transition):
        """Forward the transition to the wrapped memory, leaving the bias untouched."""
        return self.wrapped.add_transition(transition)

    def add_episode(self, episode: Episode):
        """Forward the episode to the wrapped memory, leaving the bias untouched."""
        return self.wrapped.add_episode(episode)

    def clear(self):
        return self.wrapped.clear()

    def __len__(self) -> int:
        return self.n_bias + len(self.wrapped)

    def __getitem__(self, index: int) -> T:
        if index < self.n_bias:
            return self._memory[index]
        return self.wrapped[index - self.n_bias]

    def can_sample(self, batch_size: int) -> bool:
        """
        Only count the wrapped memory, since the bias is available from the very first time step.

        Taking the bias into account would let a trainer start its updates at time step 0 on
        demonstrations alone, whereas an unbiased trainer has to collect a whole batch first.

        @ai-generated
        """
        return self.wrapped.can_sample(batch_size)

    def sample(self, batch_size: int):
        probs = np.ones(len(self))
        probs[: self.n_bias] *= self.factor
        probs /= probs.sum()
        indices = np.random.choice(range(len(self)), batch_size, replace=False, p=probs)
        return self.get_batch(indices)

    def make_batch(self, items: Iterable[T]):
        return self.wrapped.make_batch(items)

    @staticmethod
    def from_transitions(transitions: Iterable[Transition], max_size: int, factor: float = 1.0):
        transitions = list(transitions)
        memory = TransitionMemory(max_size=max_size - len(transitions))
        return BiasedMemory(transitions, memory, factor=factor)

    @staticmethod
    def from_episodes(episodes: Iterable[Episode], max_size: int, factor: float = 1.0):
        episodes = list(episodes)
        return BiasedMemory(episodes, EpisodeMemory(max_size - len(episodes)), factor=factor)
