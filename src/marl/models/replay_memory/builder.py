from typing import Literal

from .replay_memory import ReplayMemory, TransitionMemory, EpisodeMemory
from .nstep_memory import NStepMemory
from .prioritized_memory import PrioritizedMemory

class MemoryBuilder:
    def __init__(self, max_size: int, memory_type: Literal["transition", "episode"]):
        self.memory_type = memory_type
        match memory_type:
            case "transition": self.memory = TransitionMemory(max_size)
            case "episode": self.memory = EpisodeMemory(max_size)
            case other: raise ValueError(f"Unknown memory kind: {other}")

    def nstep(self, n: int, gamma: float):
        assert self.memory_type == "transition", "NStep memory is currently only implemented for transition memory"
        self.memory = NStepMemory(self.memory, n, gamma)
        return self

    def prioritized(self, alpha=0.7, beta=0.4, eps=1e-2):
        self.memory = PrioritizedMemory(self.memory, alpha, beta)
        return self
    
    def build(self) -> ReplayMemory:
        return self.memory
