from typing import TypeVar

from ..batch import Batch
from .replay_memory import ReplayMemory

T = TypeVar("T")

class MemoryWrapper(ReplayMemory[T]):
    def __init__(self, wrapped: ReplayMemory[T]) -> None:
        super().__init__(0)
        self.wrapped = wrapped

    def add(self, item: T):
        return self.wrapped.add(item)
    
    def update(self, indices: list[int], qvalues, qtargets):
        return self.wrapped.update(indices, qvalues, qtargets)
    
    def sample(self, batch_size: int) -> Batch:
        return self.wrapped.sample(batch_size)
    
    @property
    def max_size(self):
        return self.wrapped.max_size
    
    def get_batch(self, indices: list[int]) -> Batch:
        return self.wrapped.get_batch(indices)
    
    def __len__(self) -> int:
        return len(self.wrapped)
    
    def __getitem__(self, index: int) -> T:
        return self.wrapped[index]
