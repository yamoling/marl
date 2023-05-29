"""
Value Decomposition Network is a Q-learning extension that applies to cooperative multi-agent environments
with discrete action spaces.
VDN optimises the sum of rewards instead of the individual rewards of each agent.
"""

import torch
from rlenv import Observation
from marl.models import EpisodeBatch, TransitionBatch
from marl.nn import loss_functions
from .rdqn import RDQN
from .dqn import DQN


class RecurrentVDN(RDQN):
    def process_batch(self, batch: EpisodeBatch) -> EpisodeBatch:
        return batch.for_rnn()
    
    def compute_loss(self, qvalues: torch.Tensor, qtargets: torch.Tensor, batch: EpisodeBatch) -> torch.Tensor:
        return loss_functions.masked_mse(qvalues, qtargets, batch)

    def compute_targets(self, batch: EpisodeBatch) -> torch.Tensor:
        next_qvalues, _ = self._qtarget.forward(batch.obs_, batch.extras_)
        next_qvalues[batch.available_actions_ == 0.0] = -torch.inf
        next_qvalues: torch.Tensor = torch.max(next_qvalues, dim=-1)[0]
        next_qvalues = next_qvalues.reshape(batch.max_episode_len, batch.size, batch.n_agents)
        next_qvalues = next_qvalues.sum(dim=-1)
        targets = batch.rewards + self.gamma * next_qvalues * (1 - batch.dones)
        return targets

    def _sample(self) -> EpisodeBatch:
        return self.memory.sample(self._batch_size).for_rnn()

    def compute_qvalues(self, data: EpisodeBatch | Observation) -> torch.Tensor:
        qvalues = super().compute_qvalues(data)
        if isinstance(data, EpisodeBatch):
            qvalues = qvalues.sum(dim=-1)
        return qvalues
    

class LinearVDN(DQN):
    def process_batch(self, batch: TransitionBatch) -> TransitionBatch:
        return batch
    
    def compute_targets(self, batch: TransitionBatch) -> torch.Tensor:
        next_qvalues = self._qtarget.forward(batch.obs_, batch.extras_)
        next_qvalues[batch.available_actions_ == 0.0] = -torch.inf
        next_qvalues: torch.Tensor = torch.max(next_qvalues, dim=-1)[0]
        next_qvalues = next_qvalues.sum(dim=-1)
        targets = batch.rewards + self.gamma * next_qvalues * (1 - batch.dones)
        return targets

    def _sample(self) -> TransitionBatch:
        return self.memory.sample(self._batch_size)

    def compute_qvalues(self, data: TransitionBatch | Observation) -> torch.Tensor:
        qvalues = super().compute_qvalues(data)
        if isinstance(data, TransitionBatch):
            qvalues = qvalues.sum(dim=-1)
        return qvalues
    