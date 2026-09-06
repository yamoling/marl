from collections import deque
from copy import deepcopy

from marlenv import Transition

from .replay_memory import TransitionMemory


class NStepMemory(TransitionMemory):
    def __init__(self, max_size: int, n: int, gamma: float) -> None:
        super().__init__(max_size)
        assert n > 0
        self._n = n
        self._gamma = gamma
        self._pending: deque[Transition] = deque()

    def add(self, item: Transition):
        """Finalize n-step transitions and flush shortened tails at episode ends. @ai-generated"""
        self._pending.append(deepcopy(item))
        if len(self._pending) >= self._n:
            self._finalize()
        if item.is_terminal:
            while self._pending:
                self._finalize()

    def _finalize(self):
        """Store a complete reward sum, successor and matching bootstrap discount. @ai-generated"""
        first = deepcopy(self._pending[0])
        first.reward = sum(self._gamma**i * t.reward for i, t in enumerate(self._pending))
        last = self._pending[-1]
        first.next_obs = last.next_obs
        first.next_state = last.next_state
        first.done = last.done
        first.truncated = last.truncated
        first["bootstrap_discount"] = self._gamma ** len(self._pending)
        super().add(first)
        self._pending.popleft()

    def add_transition(self, transition: Transition):
        return self.add(transition)

    def clear(self):
        """Clear completed samples and the unfinished trajectory. @ai-generated"""
        super().clear()
        self._pending.clear()
