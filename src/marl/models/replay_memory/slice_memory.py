from rlenv import Transition
from ..batch import Batch
from .replay_memory import ReplayMemory



class TransitionSliceMemory(ReplayMemory[Transition]):
    def __init__(self, max_size: int, slice_size: int) -> None:
        super().__init__(max_size)
        self._slice_size = slice_size

    def __len__(self) -> int:
        return max(0, super().__len__() - self._slice_size)

    def _get_batch(self, indices: list[int]) -> Batch:
        samples = []
        for index in indices:
            transitions = [self._memory[index + s] for s in range(self._slice_size)]
            samples.append(transitions)
        return Batch.from_transition_slices(samples)

