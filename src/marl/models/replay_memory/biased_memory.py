from typing import Literal
from .replay_memory import ReplayMemory


class BiasedMemory[T](ReplayMemory[T]):
    def __init__(self, bias: list[T], max_size: int, update_on: Literal["transition", "episode"]):
        super().__init__(max_size - len(bias), update_on)
        self.bias = bias
        self.n_bias = len(bias)

    def __len__(self) -> int:
        return super().__len__() + self.n_bias

    def __getitem__(self, index: int) -> T:
        if index < self.n_bias:
            return self.bias[index]
        return super().__getitem__(index - self.n_bias)
