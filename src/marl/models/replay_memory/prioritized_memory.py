from typing import Optional
from sumtree import SumTree
import torch
from dataclasses import dataclass
from .replay_memory import ReplayMemory, T, B
from marl.models import Batch
from marl.utils import Schedule


@dataclass
class PrioritizedMemory(ReplayMemory[B, T]):
    """
    Prioritized Experience Replay.
    This class is a decorator around any other Replay Memory type.

    Credits: https://github.com/Howuhh/prioritized_experience_replay
    Paper: https://arxiv.org/abs/1511.05952
    """

    memory: ReplayMemory[B, T]
    alpha: Schedule
    beta: Schedule
    eps: float
    td_error_clipping: Optional[float]
    """Clip the TD errors to avoid numerical instability. Often required in sparse reward environments."""

    def __init__(
        self,
        memory: ReplayMemory[B, T],
        alpha: float | Schedule = 0.7,
        beta: float | Schedule = 0.4,
        eps: float = 1e-2,
        td_error_clipping: Optional[float] = 1.0,
    ):
        update_on = "transition" if memory.update_on_transitions else "episode"
        super().__init__(memory.max_size, update_on)
        self.memory = memory
        self.tree = SumTree(self.max_size)
        self.eps = eps
        self.max_priority = eps  # Initialize the max priority with epsilon
        self.td_error_clipping = td_error_clipping
        self.sampled_indices = list[int]()
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

    def sample(self, batch_size: int):
        # Sample the indices from the sumtree, proportional to their priority
        self.sampled_indices, priorities = self.tree.sample(batch_size)

        # Retrieve batch corresponding to the indices from the wrapped memory
        batch = self.memory.get_batch(self.sampled_indices)

        # Then do the book-keeping to compute the importance sampling weights
        # Concretely, we define the probability of sampling transition i as P(i) = p_i^α / \sum_{k} p_k^α
        # where p_i > 0 is the priority of transition i. (Section 3.3)

        # Personal note: at each update step, the new value of p_i^α is computed, and stored in the sumtree.
        # Therefore, the priority sampled is already p_i^α, and we only need to divide it by the sum of all
        # priorities to get the probability P(i).
        probs = torch.tensor(priorities, dtype=torch.float32) / self.tree.total

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
        batch.importance_sampling_weights = weights / torch.max(weights)
        return batch

    def get_batch(self, indices: list[int]):
        return self.memory.get_batch(indices)

    def __len__(self) -> int:
        return len(self.memory)

    def __getitem__(self, idx: int) -> T:
        return self.memory[idx]

    def update(self, td_error: torch.Tensor) -> dict[str, float]:
        # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
        # where eps is a small positive constant that prevents the edge-case of transitions not being
        # revisited once their error is zero. (Section 3.3)
        self.beta.update()
        self.alpha.update()
        with torch.no_grad():
            td_error = torch.abs(td_error)
            # Clip the TD errors to avoid numerical instability (Section 4, second §)
            if self.td_error_clipping is not None:
                td_error = torch.clip(td_error, max=self.td_error_clipping)
            priorities = (td_error + self.eps) ** self.alpha
            self.max_priority = max(self.max_priority, priorities.max().item())
        self.tree.update_batched(self.sampled_indices, priorities.cpu().tolist())
        return {
            "mean-priority": priorities.mean().item(),
            "per-alpha": self.alpha.value,
            "per-beta": self.beta.value,
        }
