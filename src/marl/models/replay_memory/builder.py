from typing import Literal

from .replay_memory import ReplayMemory, TransitionMemory, EpisodeMemory
from .nstep_memory import NStepReturnMemory
from .prioritised_memory import PrioritizedMemory

class MemoryBuilder:
    def __init__(self, max_size: int, kind: Literal["transition", "episode"]):
        self.kind = kind
        match kind:
            case "transition": self.memory = TransitionMemory(max_size)
            case "episode": self.memory = EpisodeMemory(max_size)
            case other: raise ValueError(f"Unknown memory kind: {other}")

    def nstep(self, n: int, gamma: float):
        assert self.kind == "transition", "Can currently only use nstep return with transition memory"
        self.memory = NStepReturnMemory(self.memory, n, gamma)
        return self

    def prioritized(self, alpha=0.7, beta=0.4, eps=1e-2):
        self.memory = PrioritizedMemory(self.memory, alpha, beta)
        return self
    
    def build(self) -> ReplayMemory:
        return self.memory
