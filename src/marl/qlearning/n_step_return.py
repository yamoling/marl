import torch
from marl.models import Batch
from .dqn import DQN

class NStepReturn(DQN):
    def __init__(self, n: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n

    def for_nstep_return(self, batch: Batch) -> Batch:
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
            j = self.n - 1
            while mask[j] == 0. and j > 0:
                j -= 1
            obs_.append(batch.obs_[i, j])
            extras_.append(batch.extras_[i, j])
            available_actions_.append(batch.available_actions_[i, j])
        batch.obs_ = torch.stack(obs_)
        batch.extras_ = torch.stack(extras_)
        batch.available_actions_ = torch.stack(available_actions_)
        return batch

    def _sample(self) -> Batch:
        batch = self.memory.sample(self.batch_size)
        batch = self.for_nstep_return(batch)
        batch = batch.for_individual_learners()
        return batch

