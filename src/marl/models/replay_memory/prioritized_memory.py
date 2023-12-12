from typing import Optional
from sumtree import SumTree
import torch
from dataclasses import dataclass
from .replay_memory import ReplayMemory, T
from marl.models import Batch
from marl.utils import Schedule


@dataclass
class PrioritizedMemory(ReplayMemory[T]):
    """
    Prioritized Experience Replay.
    This class is a decorator around any other Replay Memory type.

    Credits: https://github.com/Howuhh/prioritized_experience_replay
    Paper: https://arxiv.org/abs/1511.05952
    """

    memory: ReplayMemory[T]
    alpha: Schedule
    beta: Schedule
    eps: float
    priority_clipping: Optional[float]
    """Clip the priorities to avoid numerical instability. Often required in sparse reward environments."""

    def __init__(
        self,
        memory: ReplayMemory[T],
        alpha: float | Schedule = 0.7,
        beta: float | Schedule = 0.4,
        eps: float = 1e-2,
        priority_clipping: Optional[float] = None,
    ):
        super().__init__(memory.max_size)
        self.memory = memory
        self.tree = SumTree(self.max_size)
        self.eps = eps
        self.max_priority = eps  # Initialize the max priority with epsilon
        self.priority_clipping = priority_clipping
        match alpha:
            case float():
                self.alpha = Schedule.constant(alpha)
            case Schedule():
                self.alpha = alpha
            case other:
                raise ValueError(f"alpha must be a float or a Schedule, got {other}")
        match beta:
            case float():
                self.beta = Schedule.constant(beta)
            case Schedule():
                self.beta = beta
            case other:
                raise ValueError(f"beta must be a float or a Schedule, got {other}")

    def add(self, item: T):
        self.tree.add(self.max_priority)
        self.memory.add(item)

    def sample(self, batch_size: int) -> Batch:
        sample_idxs, priorities = self.tree.sample(batch_size)

        # Concretely, we define the probability of sampling transition i as P(i) = p_i^α / \sum_{k} p_k^α
        # where p_i > 0 is the priority of transition i. (Section 3.3)
        priorities = torch.tensor(priorities, dtype=torch.float32)
        probs = priorities / self.tree.total

        # The estimation of the expected value with stochastic updates relies on those updates corresponding
        # to the same distribution as its expectation. Prioritized replay introduces bias because it changes this
        # distribution in an uncontrolled fashion, and therefore changes the solution that the estimates will
        # converge to (even if the policy and state distribution are fixed). We can correct this bias by using
        # importance-sampling (IS) weights w_i = (1/N * 1/P(i))^β that fully compensates for the non-uniform
        # probabilities P(i) if β = 1. These weights can be folded into the Q-learning update by using w_i * δ_i
        # instead of δ_i (this is thus weighted IS, not ordinary IS, see e.g. Mahmood et al., 2014).
        # For stability reasons, we always normalize weights by 1/maxi wi so that they only scale the
        # update downwards (Section 3.4, first paragraph)
        weights = (len(self) * probs) ** -self.beta

        # As mentioned in Section 3.4, whenever importance sampling is used, all weights w_i were scaled
        # so that max_i w_i = 1. We found that this worked better in practice as it kept all weights
        # within a reasonable range, avoiding the possibility of extremely large updates. (Appendix B.2.1, Proportional prioritization)
        weights = weights / torch.max(weights)

        batch = self.get_batch(sample_idxs)
        batch.importance_sampling_weights = weights
        return batch

    def get_batch(self, indices: list[int]) -> Batch:
        return self.memory.get_batch(indices)

    def __len__(self) -> int:
        return len(self.memory)

    def __getitem__(self, idx: int) -> T:
        return self.memory[idx]

    def update(self, batch: Batch, td_error: torch.Tensor):
        # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
        # where eps is a small positive constant that prevents the edge-case of transitions not being
        # revisited once their error is zero. (Section 3.3)
        self.beta.update()
        self.alpha.update()
        with torch.no_grad():
            priorities = torch.abs(td_error)
            priorities = (priorities + self.eps) ** self.alpha.value
            # clip the priorities to avoid numerical instability.
            if self.priority_clipping is not None:
                priorities = torch.clip(priorities, max=self.priority_clipping)
            self.max_priority = max(self.max_priority, priorities.max().item())
        priorities = priorities.cpu()
        self.tree.update_batched(batch.sample_indices, priorities.tolist())
