from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from marl.utils import Serializable

if TYPE_CHECKING:
    from marl.models import ReplayMemory


@dataclass
class MemoryConfig(Serializable):
    kind: Literal["episode", "transition"]
    max_size: int

    def make(self) -> ReplayMemory:
        from marl.models.replay_memory import EpisodeMemory, TransitionMemory

        match self.kind:
            case "episode":
                memory = EpisodeMemory(self.max_size)
            case "transition":
                memory = TransitionMemory(self.max_size)
            case other:
                raise ValueError(f"Unknown memory kind: {other}")
        return memory


@dataclass
class PERConfig(MemoryConfig):
    alpha: float = 0.7
    beta: float = 0.4
    eps: float = 1e-2
    td_error_clipping: float | None = 1.0
    is_multi_objective: bool = False

    def make(self):
        from marl.models.replay_memory import PrioritizedMemory

        base_memory = super().make()
        return PrioritizedMemory(base_memory, self.is_multi_objective, self.alpha, self.beta, self.eps, self.td_error_clipping)
