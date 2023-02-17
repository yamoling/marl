import torch
from rlenv import Transition
from .batch import Batch
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


class NStepReturnMemory(TransitionSliceMemory):
    def _get_batch(self, indices: list[int]) -> Batch:
        batch = super()._get_batch(indices)
        batch.obs = batch.obs[:, 0]
        batch.extras = batch.extras[:, 0]
        batch.actions = batch.actions[:, 0]
        batch.rewards = batch.rewards * batch.masks
        batch.rewards = torch.sum(batch.rewards, dim=-1)
        dones = batch.dones * batch.masks
        batch.dones = torch.clamp(torch.sum(dones, dim=-1), max=1.)
        obs_ = []
        extras_ = []
        available_actions_ = []
        for i, mask in enumerate(batch.masks):
            j = self._slice_size - 1
            while mask[j] == 0. and j > 0:
                j -= 1
            obs_.append(batch.obs_[i, j])
            extras_.append(batch.extras_[i, j])
            available_actions_.append(batch.available_actions_[i, j])
        batch.obs_ = torch.stack(obs_)
        batch.extras_ = torch.stack(extras_)
        batch.available_actions_ = torch.stack(available_actions_)
        return batch