from typing import List, TypeVar
from sumtree import SumTree
import torch

from .batch import Batch
from .replay_memory import ReplayMemory


T = TypeVar("T")


class PrioritizedMemory(ReplayMemory[T]):
    """
    Prioritized Experience Replay.
    This class is a decorator around any other Replay Memory type.

    Credits: https://github.com/Howuhh/prioritized_experience_replay
    Paper: https://arxiv.org/abs/1511.05952
    """

    def __init__(self, memory: ReplayMemory[T], alpha: float, beta: float, eps: float = 1e-2) -> None:
        super().__init__(0)
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
        sample_idxs = []
        priorities = []
        # To sample a minibatch of size k, the range [0, p_total] is divided equally into k ranges.
        # Next, a value is uniformly sampled from each range. Finally the transitions that correspond
        # to each of these sampled values are retrieved from the tree. (Appendix B.2.1, Proportional prioritization)
        segment_size = self._tree.total / batch_size
        values = torch.rand(batch_size) * segment_size + torch.arange(0, batch_size) * segment_size
        max_idx = len(self._wrapped_memory) - 1
        for cumsum in values:
            # sample_idx is a sample index in buffer, needed further to sample actual transitions
            # tree_idx is a index of a sample in the tree, needed further to update priorities
            idx, priority = self._tree.get(cumsum)
            idx = min(max_idx, idx)
            priorities.append(priority)
            sample_idxs.append(idx)

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
        weights = (len(self._wrapped_memory) * probs) ** -self._beta

        # As mentioned in Section 3.4, whenever importance sampling is used, all weights w_i were scaled
        # so that max_i w_i = 1. We found that this worked better in practice as it kept all weights
        # within a reasonable range, avoiding the possibility of extremely large updates. (Appendix B.2.1, Proportional prioritization)
        weights = weights / weights.max()

        batch = self._get_batch(sample_idxs)
        batch.is_weights = weights.unsqueeze(-1)
        batch.sample_index = sample_idxs
        return batch

    def _get_batch(self, indices: List[int]) -> Batch:
        return self._wrapped_memory._get_batch(indices)

    def __len__(self) -> int:
        return len(self._wrapped_memory)
    
    def __getitem__(self, idx: int) -> T:
        return self._wrapped_memory[idx]

    def update(self, indices: List[int], qvalues: torch.Tensor, qtargets: torch.Tensor):
        with torch.no_grad():
            priorities = (qtargets - qvalues).abs()
            priorities = priorities.sum(dim=-1)
            priorities = (priorities + self._eps) ** self._alpha
            self._max_priority = max(self._max_priority, priorities.max().item())
        priorities = priorities.cpu()
        for idx, priority in zip(indices, priorities):
            # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
            # where eps is a small positive constant that prevents the edge-case of transitions not being
            # revisited once their error is zero. (Section 3.3)
            self._tree.update(idx, priority.item())
