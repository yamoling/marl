from collections import deque
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, field

import torch
from marlenv import Transition

from marl.models.batch import TransitionBatch
from marl.utils import tuning

from .replay_memory import TransitionMemory


@dataclass
class NStepMemory(TransitionMemory):
    n: int = field(metadata=tuning(2, 5))
    gamma: float

    def __post_init__(self) -> None:
        """@ai-edited"""
        super().__post_init__()
        self._pending = deque[Transition]()
        self._num_finalized = 0

    def add(self, item: Transition):
        """Finalize n-step transitions and flush shortened tails at episode ends. @ai-generated"""
        self._pending.append(deepcopy(item))
        # As soon as we have `n` items in the memory, link the 0th with the nth
        if len(self._pending) >= self.n:
            self._finalize()
        if item.is_terminal:
            # Flush
            while len(self._pending) > 0:
                self._finalize()

    def _finalize(self):
        """Store a transition with its accumulated reward, successor and bootstrap discount. @ai-generated"""
        first = deepcopy(self._pending[0])
        for i, transition in enumerate(self._pending):
            if i > 0:
                first.reward += self.gamma**i * transition.reward
        last = self._pending[-1]
        first.next_obs = last.next_obs
        first.next_state = last.next_state
        first.done = last.done
        first.truncated = last.truncated
        first["n-step-gamma"] = self.gamma ** len(self._pending)
        super().add(first)
        self._num_finalized += 1
        self._pending.popleft()

    def make_batch(self, items: Iterable[Transition]) -> TransitionBatch:
        batch = super().make_batch(items)
        batch.gamma = batch["n-step-gamma"].to(torch.float32)
        return batch

    def add_transition(self, transition: Transition):
        return self.add(transition)

    def clear(self):
        """Clear completed samples and the unfinished trajectory. @ai-generated"""
        super().clear()
        self._pending.clear()
        self._num_finalized = 0
