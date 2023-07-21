from copy import deepcopy
from rlenv import Transition
from .replay_memory import TransitionMemory


class NStepMemory(TransitionMemory):
    def __init__(self, max_size: int, n: int, gamma: float) -> None:
        super().__init__(max_size)
        assert n > 0
        self._n = n
        self._gamma = gamma
        self._episode_len = 0

    def __len__(self) -> int:
        return max(0, len(self._memory) - self._n)

    def add(self, item: Transition):
        item = deepcopy(item)
        self._episode_len += 1
        r = item.reward
        # Update the rewards of the last n transitions backward
        for i in range(1, min(self._n, self._episode_len)):
            r = self._gamma * r
            self._memory[-i].reward += r

        # Add the new transition to the memory
        super().add(item)
        if item.is_terminal:
            # Update the last n observations such that their next obs is the one of the episode
            for i in range(2, min(self._n, self._episode_len) + 1):
                self._update_transition(-i, item.obs_, item.done, item.truncated)
            self._episode_len = 0

    def _update_transition(self, index: int, obs_, done, truncated):
        t = self._memory[index]
        t.obs_ = obs_
        t.done = done
        t.truncated = truncated

    def summary(self):
        return {
            **super().summary(),
            "n": self._n,
            "gamma": self._gamma,
        }
