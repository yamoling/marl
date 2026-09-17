from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal, Self, override

import numpy as np
from marlenv import Episode, Transition

from marl.utils import PickleArtifact

from .replay_memory import ReplayMemory


@dataclass
class BiasedMemory[T](ReplayMemory[T]):
    """Replay memory with immutable demonstrations stored in an external pickle artifact."""

    demonstrations: PickleArtifact[list[Episode]]
    wrapped: ReplayMemory[T]
    n_bias: int
    factor: float = 1.0
    """Factor that multiplies the probability of sampling biased items."""
    max_size: int = field(init=False)
    update_on: Literal["episode", "transition"] = field(init=False)
    _bias: list[T] | None = field(init=False, default=None, repr=False)

    def __post_init__(self):
        """Initialize lightweight metadata without loading the demonstration artifact. @ai-edited"""
        self.max_size = self.wrapped.max_size
        self.update_on = self.wrapped.update_on
        assert self.n_bias < self.max_size, "The bias should be smaller than the memory size"
        assert self.n_bias > 0, "There should be at least one element to bias towards"
        assert self.factor > 0, "factor must be greater than 0"
        super().__post_init__()

    @classmethod
    def from_episodes(cls, episodes: Iterable[Episode], wrapped: ReplayMemory[T], factor: float = 1.0) -> Self:
        """Create a biased memory from canonical episodes and defer their persistence. @ai-generated"""
        episodes = list(episodes)
        if wrapped.update_on_transitions:
            n_bias = sum(len(episode) for episode in episodes)
        else:
            n_bias = len(episodes)
        artifact = PickleArtifact.create(episodes, count=len(episodes))
        return cls(artifact, wrapped, n_bias, factor)

    @property
    def bias(self) -> list[T]:
        """Materialize and process-locally cache items matching the wrapped memory type. @ai-generated"""
        if self._bias is None:
            episodes = self.demonstrations.load()
            if self.update_on_transitions:
                items = [transition for episode in episodes for transition in episode.transitions()]
            else:
                items = episodes
            if len(items) != self.n_bias:
                raise ValueError(f"Expected {self.n_bias} biased items, loaded {len(items)}.")
            self._bias = items
        return self._bias

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
            return self.bias[index]
        return self.wrapped[index - self.n_bias]

    @override
    def can_sample(self, batch_size: int) -> bool:
        return self.wrapped.can_sample(batch_size)

    @override
    def sample(self, batch_size: int):
        probs = np.ones(len(self))
        probs[: self.n_bias] *= self.factor
        probs /= probs.sum()
        indices = np.random.choice(range(len(self)), batch_size, replace=False, p=probs)
        return self.get_batch(indices)

    @override
    def make_batch(self, items: Iterable[T]):
        return self.wrapped.make_batch(items)
