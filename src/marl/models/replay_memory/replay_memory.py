from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Deque, Generic, Iterable, TypeVar

import numpy as np
import torch
from rlenv import Episode, Transition

from marl.models.batch import Batch, EpisodeBatch, TransitionBatch


T = TypeVar("T")


@dataclass
class ReplayMemory(Generic[T], ABC):
    """Parent class of any ReplayMemory"""

    max_size: int

    def __post_init__(self):
        self._memory: Deque[T] = deque(maxlen=self.max_size)

    def add(self, item: T):
        """Add an item (transition, episode, ...) to the memory"""
        self._memory.append(item)

    def update(self, batch: Batch, qvalues: torch.Tensor, qtargets: torch.Tensor):
        """Update the data in the memory"""

    def sample(self, batch_size: int) -> Batch:
        """Sample the memory to retrieve a `Batch`"""
        indices = np.random.randint(0, len(self), batch_size)
        return self.get_batch(indices)

    def clear(self):
        self._memory.clear()

    @abstractmethod
    def get_batch(self, indices: Iterable[int]) -> Batch:
        """Create a `Batch` from the given indices"""

    def __len__(self) -> int:
        return len(self._memory)

    def __getitem__(self, index: int) -> T:
        return self._memory[index]


class TransitionMemory(ReplayMemory[Transition]):
    """Replay Memory that stores Transitions"""

    def get_batch(self, indices: Iterable[int]) -> TransitionBatch:
        transitions = [self._memory[i] for i in indices]
        return TransitionBatch(transitions, list(indices))


class EpisodeMemory(ReplayMemory[Episode]):
    """Replay Memory that stores and samples full Episodes"""

    def get_batch(self, indices: list[int]) -> EpisodeBatch:
        episodes = [self._memory[i] for i in indices]
        return EpisodeBatch(episodes, indices)
