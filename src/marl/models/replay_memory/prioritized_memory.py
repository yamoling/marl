from typing import List, TypeVar
from sumtree import SumTree
import torch

from ..batch import Batch
from .replay_memory import ReplayMemory


T = TypeVar("T")


class PrioritizedMemory(ReplayMemory[T]):
    """
    Prioritized Experience Replay.
    This class is a decorator around any other Replay Memory type.

    Credits: https://github.com/Howuhh/prioritized_experience_replay
    Paper: https://arxiv.org/abs/1511.05952
    """

    def __init__(self, memory: ReplayMemory[T], alpha=0.7, beta=0.4, eps: float = 1e-2):
        super().__init__(memory.max_size)
        self._wrapped_memory = memory
        self._tree = SumTree(memory.max_size)
        self._max_priority = eps  # Initialize the max priority with epsilon
        self._eps = eps
        self._alpha = alpha
        self._beta = beta

    def add(self, item: T):
        self._tree.add(self._max_priority)
        self._wrapped_memory.add(item)

    def sample(self, batch_size: int) -> Batch:
        sample_idxs, priorities = self._tree.sample(batch_size)

        # Concretely, we define the probability of sampling transition i as P(i) = p_i^α / \sum_{k} p_k^α
        # where p_i > 0 is the priority of transition i. (Section 3.3)
        priorities = torch.tensor(priorities, dtype=torch.float32)
        probs = priorities / self._tree.total

        # The estimation of the expected value with stochastic updates relies on those updates corresponding
        # to the same distribution as its expectation. Prioritized replay introduces bias because it changes this
        # distribution in an uncontrolled fashion, and therefore changes the solution that the estimates will
        # converge to (even if the policy and state distribution are fixed). We can correct this bias by using
        # importance-sampling (IS) weights w_i = (1/N * 1/P(i))^β that fully compensates for the non-uniform
        # probabilities P(i) if β = 1. These weights can be folded into the Q-learning update by using w_i * δ_i
        # instead of δ_i (this is thus weighted IS, not ordinary IS, see e.g. Mahmood et al., 2014).
        # For stability reasons, we always normalize weights by 1/maxi wi so that they only scale the
        # update downwards (Section 3.4, first paragraph)
        weights = (len(self) * probs) ** -self._beta

        # As mentioned in Section 3.4, whenever importance sampling is used, all weights w_i were scaled
        # so that max_i w_i = 1. We found that this worked better in practice as it kept all weights
        # within a reasonable range, avoiding the possibility of extremely large updates. (Appendix B.2.1, Proportional prioritization)
        weights = weights / weights.max()

        batch = self.get_batch(sample_idxs)
        batch.importance_sampling_weights = weights
        return batch

    def get_batch(self, indices: List[int]) -> Batch:
        return self._wrapped_memory.get_batch(indices)

    def __len__(self) -> int:
        return len(self._wrapped_memory)
    
    def __getitem__(self, idx: int) -> T:
        return self._wrapped_memory[idx]

    def update(self, batch: Batch, qvalues: torch.Tensor, qtargets: torch.Tensor):
        # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
        # where eps is a small positive constant that prevents the edge-case of transitions not being
        # revisited once their error is zero. (Section 3.3)
        with torch.no_grad():
            priorities = torch.abs(qtargets - qvalues)
            priorities = (priorities + self._eps) ** self._alpha
            self._max_priority = max(self._max_priority, priorities.max().item())
        priorities = priorities.cpu()
        for idx, priority in zip(batch.sample_indices, priorities):
            self._tree.update(idx, priority.item())

    def summary(self):
        summary = super().summary()
        summary.pop("max_size")
        return {
            **summary,
            "alpha": self._alpha,
            "beta": self._beta,
            "eps": self._eps,
            "memory": self._wrapped_memory.summary()
        }
    
    @classmethod
    def from_summary(cls, summary: dict[str, ]):
        from marl.models import replay_memory
        summary["memory"] = replay_memory.from_summary(summary.pop("memory"))
        return super().from_summary(summary)