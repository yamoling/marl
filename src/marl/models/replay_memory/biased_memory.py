from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal, Self, override

import numpy as np
from marlenv import Episode, Transition

from marl.utils.marlenv_deserialization import episode_from_dict, transition_from_dict

from .replay_memory import EpisodeMemory, ReplayMemory, TransitionMemory


@dataclass
class BiasedMemory[T](ReplayMemory[T]):
    bias: list[T]
    wrapped: ReplayMemory[T]
    factor: float = 1.0
    """Factor that multiplies the probability of sampling biased items."""
    max_size: int = field(init=False)
    update_on: Literal["episode", "transition"] = field(init=False)

    def __post_init__(self):
        self.max_size = self.wrapped.max_size
        self.update_on = self.wrapped.update_on
        assert len(self.bias) < self.max_size, "The bias should be smaller than the memory size"
        assert len(self.bias) > 0, "There sould be at least one element to bias towards"
        assert self.factor > 0, "factor must be greater than 0"
        super().__post_init__()
        self._memory.extend(self.bias)
        self.n_bias = len(self.bias)

    @classmethod
    def from_dict(cls, d: dict[str, Any], *, exact_type: bool = False) -> Self:
        """Decode bias items according to the restored wrapped memory. @ai-generated"""
        wrapped = ReplayMemory.from_dict(d["wrapped"])
        d["wrapped"] = wrapped
        if wrapped.update_on_transitions:
            d["bias"] = [transition_from_dict(item) for item in d["bias"]]
        else:
            d["bias"] = [episode_from_dict(item) for item in d["bias"]]
        return super().from_dict(d, exact_type=exact_type)

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
            return self._memory[index]
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

    @staticmethod
    def from_transitions(transitions: Iterable[Transition], max_size: int, factor: float = 1.0):
        transitions = list(transitions)
        memory = TransitionMemory(max_size=max_size - len(transitions))
        return BiasedMemory(transitions, memory, factor=factor)

    @staticmethod
    def from_episodes(episodes: Iterable[Episode], max_size: int, factor: float = 1.0):
        episodes = list(episodes)
        return BiasedMemory(episodes, EpisodeMemory(max_size - len(episodes)), factor=factor)
