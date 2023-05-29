from copy import deepcopy
from rlenv import Transition
from .replay_memory import TransitionMemory


class NStepMemory(TransitionMemory):
    def __init__(self, max_size: int, n: int, gamma: float) -> None:
        super().__init__(max_size)
        self._n = n
        self._gamma = gamma
        self._episode_len = 0

    def __len__(self) -> int:
        return max(0, len(self._memory) - self._n)

    def add(self, item: Transition):
        item = deepcopy(item)
        self._episode_len += 1
        r = item.reward
        # Update the rewards of the last n transitions
        for i in range(1, min(self._n, self._episode_len)):
            r = self._gamma * r
            self._memory[-i].reward += r
        # Add the transition to the memory
        super().add(item)
        # Update the next obs of the -n transition to be the current one
        if self._episode_len >= self._n:
            t = self._memory[-self._n]
            t.obs_ = item.obs
        if item.is_terminal:
            self._episode_len = 0
            # Update the last n observations such that their next obs is the one of the episode
            for i in range(-self._n+1, -1):
                self._memory[i].obs_ = item.obs_
                self._memory[i].done = item.done

    def summary(self):
        return {
            **super().summary(),
            "n": self._n,
            "gamma": self._gamma,
        }
