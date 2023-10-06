from typing import List, Any
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
    alpha: Schedule = 0.7
    beta: Schedule = 0.4
    eps: float = 1e-2

    def __init__(self, memory: ReplayMemory[T], alpha: float | Schedule = 0.7, beta: float | Schedule = 0.4, eps: float = 1e-2):
        super().__init__(memory.max_size)
        if isinstance(self.alpha, float):
            alpha = Schedule.constant(self.alpha)
        if isinstance(self.beta, float):
            beta = Schedule.constant(self.beta)
        self.memory = memory
        self.tree = SumTree(self.max_size)
        self.eps = eps
        self.max_priority = eps  # Initialize the max priority with epsilon
        self.alpha = alpha
        self.beta = beta

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

    def get_batch(self, indices: List[int]) -> Batch:
        return self.memory.get_batch(indices)

    def __len__(self) -> int:
        return len(self.memory)

    def __getitem__(self, idx: int) -> T:
        return self.memory[idx]

    def update(self, batch: Batch, qvalues: torch.Tensor, qtargets: torch.Tensor):
        # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
        # where eps is a small positive constant that prevents the edge-case of transitions not being
        # revisited once their error is zero. (Section 3.3)
        self.beta.update()
        self.alpha.update()
        with torch.no_grad():
            priorities = torch.abs(qtargets - qvalues)
            priorities = (priorities + self.eps) ** self.alpha
            self.max_priority = max(self.max_priority, priorities.max().item())
        priorities = priorities.cpu()
        for idx, priority in zip(batch.sample_indices, priorities):
            self.tree.update(idx, priority.item())

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        from marl.models import replay_memory
        from marl.utils import schedule

        data["memory"] = replay_memory.load(data["memory"])
        data["alpha"] = schedule.from_dict(data["alpha"])
        data["beta"] = schedule.from_dict(data["beta"])
        try:
            data.pop("max_size")
        except KeyError:
            pass
        return super().from_dict(data)
