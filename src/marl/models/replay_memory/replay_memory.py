from abc import ABC, abstractmethod
from collections import deque
from typing import Generic, TypeVar, Deque
from rlenv import Episode, Transition
import numpy as np
import torch

from marl.models.batch import Batch
from marl.models.batch import TransitionsBatch, EpisodeBatch
from marl.utils.summarizable import Summarizable


T = TypeVar("T")
B = TypeVar("B", bound=Batch)


class ReplayMemory(Summarizable, Generic[T, B], ABC):
    """Parent class of any ReplayMemory"""

    def __init__(self, max_size: int) -> None:
        self._memory: Deque[T] = deque(maxlen=max_size)
        self._max_size = max_size

    def add(self, item: T):
        """Add an item (transition, episode, ...) to the memory"""
        self._memory.append(item)

    def update(self, batch: B, qvalues: torch.Tensor, qtargets: torch.Tensor):
        """Update the data in the memory"""

    def sample(self, batch_size: int) -> B:
        """Sample the memory to retrieve a `Batch`"""
        indices = np.random.randint(0, len(self), batch_size)
        return self.get_batch(indices)
    
    def clear(self):
        self._memory.clear()
    
    @property
    def max_size(self):
        return self._max_size

    @abstractmethod
    def get_batch(self, indices: list[int]) -> Batch:
        """Create a `Batch` from the given indices"""

    def __len__(self) -> int:
        return len(self._memory)
    
    def __getitem__(self, index: int) -> T:
        return self._memory[index]

    def summary(self):
        return {
            **super().summary(),
            "max_size": self._max_size
        }
    

class TransitionMemory(ReplayMemory[Transition, TransitionsBatch]):
    """Replay Memory that stores Transitions"""

    def get_batch(
            self, 
            indices: list[int]
        ) -> TransitionsBatch:
        transitions = [self._memory[i] for i in indices]
        return TransitionsBatch(transitions, indices)


class EpisodeMemory(ReplayMemory[Episode, EpisodeBatch]):
    """Replay Memory that stores and samples full Episodes"""
    def get_batch(self, indices: list[int]) -> EpisodeBatch:
        episodes = [self._memory[i] for i in indices]
        return EpisodeBatch(episodes, indices)
