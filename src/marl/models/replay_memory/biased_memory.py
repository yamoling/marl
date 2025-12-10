from typing import Iterable
from .replay_memory import ReplayMemory, TransitionMemory
from marl.models.batch import TransitionBatch
from marl.models import Batch
from marlenv import Transition
from collections import deque
from dataclasses import dataclass


@dataclass
class BiasedMemory[T, B: Batch](ReplayMemory[T, B]):
    n_bias: int
    wrapped: ReplayMemory[T, B]

    def __init__(self, bias: Iterable[T], memory: ReplayMemory[T, B]):
        bias = list(bias)
        assert len(bias) > 0, "There sould be at least one element to bias towards"
        super().__init__(memory.max_size + len(bias), memory.updates_on)
        self._memory = deque(bias)
        self.n_bias = len(bias)
        self.wrapped = memory

    def add(self, item: T):
        return self.wrapped.add(item)

    def clear(self):
        return self.wrapped.clear()

    def __len__(self) -> int:
        return self.n_bias + len(self.wrapped)

    def __getitem__(self, index: int) -> T:
        if index < self.n_bias:
            return self._memory[index]
        return self.wrapped[index - self.n_bias]

    def get_batch(self, indices: Iterable[int]) -> B:
        wrapped_indices = [i - self.n_bias for i in indices if i >= self.n_bias]
        bias_indices = [i for i in indices if i < self.n_bias]
        batch = self.wrapped.get_batch(wrapped_indices)
        if len(bias_indices) > 0:
            bias_items = [self._memory[i] for i in bias_indices]
            batch = batch.extend(bias_items)
        return batch

    @staticmethod
    def from_transitions(transitions: Iterable[Transition], max_size: int) -> "BiasedMemory[Transition, TransitionBatch]":
        memory = TransitionMemory(max_size=max_size)
        for transition in transitions:
            memory.add(transition)
        return BiasedMemory(transitions, memory)
