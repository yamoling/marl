from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Deque, Generic, Iterable, Literal
from typing_extensions import TypeVar

import numpy as np
from marlenv import Episode, Transition

from marl.models.batch import Batch, EpisodeBatch, TransitionBatch


T = TypeVar("T")
B = TypeVar("B", bound=Batch, default=Batch)


@dataclass
class ReplayMemory(Generic[T, B], ABC):
    """Parent class of any ReplayMemory"""

    max_size: int
    name: str
    update_on_transitions: bool
    update_on_episodes: bool

    def __init__(self, max_size: int, update_on: Literal["transition", "episode"]):
        self._memory: Deque[T] = deque(maxlen=max_size)
        self.max_size = max_size
        self.name = self.__class__.__name__
        self.update_on_transitions = update_on == "transition"
        self.update_on_episodes = update_on == "episode"

    def add(self, item: T):
        """Add an item (transition, episode, ...) to the memory"""
        self._memory.append(item)

    def sample(self, batch_size: int) -> B:
        """Sample the memory to retrieve a `Batch`"""
        indices = np.random.randint(0, len(self), batch_size)
        return self.get_batch(indices)

    def can_sample(self, batch_size: int) -> bool:
        """Return whether the memory contains enough items to sample a batch of the given size"""
        return len(self) >= batch_size

    def clear(self):
        self._memory.clear()

    @property
    def is_full(self):
        return len(self) == self.max_size

    @abstractmethod
    def get_batch(self, indices: Iterable[int]) -> B:
        """Create a `Batch` from the given indices"""

    def __len__(self) -> int:
        return len(self._memory)

    def __getitem__(self, index: int) -> T:
        return self._memory[index]


@dataclass
class TransitionMemory(ReplayMemory[Transition, TransitionBatch]):
    """Replay Memory that stores Transitions"""

    def __init__(self, max_size: int):
        super().__init__(max_size, "transition")

    def get_batch(self, indices: Iterable[int]):
        transitions = [self._memory[i] for i in indices]
        return TransitionBatch(transitions)


@dataclass
class EpisodeMemory(ReplayMemory[Episode, EpisodeBatch]):
    """Replay Memory that stores and samples full Episodes"""

    def __init__(self, max_size: int):
        super().__init__(max_size, "episode")

    def get_batch(self, indices: Iterable[int]):
        episodes = [self._memory[i] for i in indices]
        return EpisodeBatch(episodes)
