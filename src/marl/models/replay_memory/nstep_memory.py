from collections import deque
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, field

from marlenv import Transition

from marl.models.batch import TransitionBatch
from marl.utils import tuning

from .replay_memory import TransitionMemory


@dataclass
class NStepMemory(TransitionMemory):
    n: int = field(metadata=tuning(2, 5))
    gamma: float

    def __post_init__(self) -> None:
        super().__post_init__()
        self._pending = deque[Transition]()
        self._episode = list[Transition]()

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
            # For each transition of the episode, adjust the gamma value
            ep_length = len(self._episode)
            for i, transition in enumerate(self._episode):
                # Most transitions have self.gamma**n, but the exponent of the last ones must be capped
                exponent = min(self.n, ep_length - i - 1)
                transition["n-step-gamma"] = self.gamma**exponent
            self._episode.clear()

    def _finalize(self):
        """Store a transition with its accumulated reward and n-step successor."""
        first = deepcopy(self._pending[0])
        for i, transition in enumerate(self._pending):
            if i > 0:
                first.reward += self.gamma**i * transition.reward
        last = self._pending[-1]
        first.next_obs = last.next_obs
        first.next_state = last.next_state
        first.done = last.done
        first.truncated = last.truncated
        super().add(first)
        self._episode.append(self._pending.popleft())

    def make_batch(self, items: Iterable[Transition]) -> TransitionBatch:
        batch = super().make_batch(items)
        batch.gamma = batch["n-step-gamma"]
        return batch

    def add_transition(self, transition: Transition):
        return self.add(transition)

    def clear(self):
        """Clear completed samples and the unfinished trajectory. @ai-generated"""
        super().clear()
        self._pending.clear()
