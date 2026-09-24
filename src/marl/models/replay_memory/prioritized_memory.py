from collections.abc import Iterable
from dataclasses import KW_ONLY, dataclass, field
from typing import Literal

import torch
from sumtree import SumTree

from marl.utils import Schedule

from .nstep_memory import NStepMemory
from .replay_memory import ReplayMemory


@dataclass
class PrioritizedMemory[T](ReplayMemory[T]):
    """
    Prioritized Experience Replay.
    This class is a decorator around any other Replay Memory type.

    Credits: https://github.com/Howuhh/prioritized_experience_replay
    Paper: https://arxiv.org/abs/1511.05952
    """

    memory: ReplayMemory[T]
    alpha: Schedule = field(default_factory=lambda: Schedule.constant(0.7))
    beta: Schedule = field(default_factory=lambda: Schedule.constant(0.4))
    eps: float = 1e-2
    td_error_clipping: float | None = 1.0
    max_size: int = field(init=False)
    _: KW_ONLY
    multi_objective: bool = False
    update_on: Literal["transition", "episode"] = field(init=False)

    def __post_init__(self):
        """Reject biased stores whose demonstration prefix has no priority slots. @ai-edited"""
        from .biased_memory import BiasedMemory

        if isinstance(self.memory, BiasedMemory):
            raise TypeError("PrioritizedMemory cannot wrap BiasedMemory: demonstration indices do not match priority tree slots.")
        self.max_size = self.memory.max_size
        self.update_on = self.memory.update_on
        super().__post_init__()
        self.sampled_indices = list[int]()
        self.tree = SumTree(self.max_size)
        self.max_priority = self.eps
        self._next_index = 0

    def add(self, item: T):
        """Advance the tree once per stored item, including n-step episode tails. @ai-generated"""
        before = self.memory._num_finalized if isinstance(self.memory, NStepMemory) else None
        self.memory.add(item)
        count = self.memory._num_finalized - before if before is not None else 1
        for _ in range(count):
            self.tree.add(self.max_priority)
        self._next_index = (self._next_index + count) % self.max_size

    def add_transition(self, transition):
        if self.update_on_transitions:
            self.add(transition)

    def add_episode(self, episode):
        if self.update_on_episodes:
            self.add(episode)

    def clear(self):
        """Reset both the replay store and its priority index. @ai-generated"""
        self.memory.clear()
        self.tree = SumTree(self.max_size)
        self._next_index = 0
        self.sampled_indices = []
        self.max_priority = self.eps

    def sample(self, batch_size: int):
        """Map physical priority slots to logical deque indices after eviction. @ai-generated"""
        # Sample the indices from the sumtree, proportional to their priority
        self.sampled_indices, priorities = self.tree.sample(batch_size)

        # Retrieve batch corresponding to the indices from the wrapped memory
        if len(self) == self.max_size:
            indices = [(i - self._next_index) % self.max_size for i in self.sampled_indices]
        else:
            indices = self.sampled_indices
        batch = self.memory.get_batch(indices)

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

    def get_batch(self, indices: Iterable[int]):
        return self.memory.get_batch(indices)

    def __len__(self) -> int:
        return len(self.memory)

    def __getitem__(self, idx: int) -> T:
        return self.memory[idx]

    def update(self, time_step: int, /, td_error: torch.Tensor | None = None, **kwargs) -> dict[str, float]:
        """Use maximum absolute error per replay item across agents/time/objectives. @ai-generated"""
        if td_error is None:
            raise ValueError("'td_error' keyword argument must be provided to update the priorities of the sampled transitions.")
        # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
        # where eps is a small positive constant that prevents the edge-case of transitions not being
        # revisited once their error is zero. (Section 3.3)
        self.beta.update(time_step)
        self.alpha.update(time_step)
        with torch.no_grad():
            td_error = torch.abs(td_error)
            if self.multi_objective:
                td_error = torch.mean(td_error, dim=-1)
            if self.update_on_episodes:
                td_error = td_error.movedim(1, 0)
            td_error = td_error.reshape(len(self.sampled_indices), -1).amax(dim=-1)
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

    def make_batch(self, items: Iterable[T]):
        return self.memory.make_batch(items)
