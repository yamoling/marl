from abc import ABC, abstractmethod
from collections import deque
from typing import Generic, TypeVar, Deque, List
from rlenv import Episode, Transition
import numpy as np

from .batch import Batch


T = TypeVar("T")


class ReplayMemory(Generic[T], ABC):
    """Parent class of any ReplayMemory"""

    def __init__(self, max_size: int) -> None:
        self._memory: Deque[T] = deque(maxlen=max_size)

    def add(self, item: T):
        """Add an item (transition, episode, ...) to the memory"""
        self._memory.append(item)

    def update(self, indices: list[int], priorities: list[float]):
        """Update the data in the memory"""

    def sample(self, batch_size: int) -> Batch:
        """Sample the memory to retrieve a `Batch`"""
        indices = np.random.randint(0, len(self), batch_size)
        return self._get_batch(indices)

    @abstractmethod
    def _get_batch(self, indices: list[int]) -> Batch:
        """Create a `Batch` from the given indices"""

    def __len__(self) -> int:
        return len(self._memory)



class TransitionMemory(ReplayMemory[Transition]):
    """Replay Memory that stores Transitions"""

    def _get_batch(self, indices: list[int]) -> Batch:
        samples = [self._memory[i] for i in indices]
        return Batch.from_transitions(samples)


class EpisodeMemory(ReplayMemory[Episode]):
    """Replay Memory that stores and samples full Episodes"""
    def _get_batch(self, indices: list[int]) -> Batch:
        episodes = [self._memory[i] for i in indices]
        return Batch.from_episodes(episodes)
