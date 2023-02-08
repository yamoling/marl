"""
Value Decomposition Network is a Q-learning extension that applies to cooperative multi-agent environments
with a discrete action space.
VDN optimises the sum of rewards instead of the individual rewards of each agent.
"""

import torch
from rlenv import Observation
from marl.models import Batch
from .rdqn import RDQN
from .dqn import DQN
from .qlearning_wrapper import DeepQWrapper


class RecurrentVDN(RDQN):
    def compute_targets(self, batch: Batch) -> torch.Tensor:
        next_qvalues, _ = self.qtarget.forward(batch.obs_, batch.extras_)
        next_qvalues[batch.available_actions_ == 0.0] = -torch.inf
        next_qvalues: torch.Tensor = torch.max(next_qvalues, dim=-1)[0]
        next_qvalues = next_qvalues.reshape(batch.max_episode_len, batch.size, batch.n_agents)
        next_qvalues = next_qvalues.sum(dim=-1)
        targets = batch.rewards + self.gamma * next_qvalues * (1 - batch.dones)
        return targets

    def _sample(self) -> Batch:
        return self.memory.sample(self.batch_size).for_rnn()

    def compute_qvalues(self, data: Batch | Observation) -> torch.Tensor:
        qvalues = super().compute_qvalues(data)
        if isinstance(data, Batch):
            qvalues = qvalues.sum(dim=-1)
        return qvalues




class VDN(DeepQWrapper):
    def __init__(self, wrapped: DQN|RDQN) -> None:
        super().__init__(wrapped)
        # Type hinting
        self.algo: DQN | RDQN = self.algo 

    def compute_qvalues(self, data: Batch | Observation) -> torch.Tensor:
        qvalues = super().compute_qvalues(data)
        if isinstance(data, Batch):
            qvalues = qvalues.sum(dim=-1)
        return qvalues

    def compute_targets(self, batch: Batch) -> torch.Tensor:
        next_qvalues = self.algo.qtarget.forward(batch.obs_, batch.extras_)
        if isinstance(next_qvalues, tuple):
            next_qvalues = next_qvalues[0]
        next_qvalues[batch.available_actions_ == 0.0] = -torch.inf
        next_qvalues: torch.Tensor = torch.max(next_qvalues, dim=-1)[0]
        next_qvalues = next_qvalues.sum(dim=-1)
        targets = batch.rewards + self.algo.gamma * next_qvalues * (1 - batch.dones)
        return targets

class LinearVDN(DQN):

    def compute_targets(self, batch: Batch) -> torch.Tensor:
        next_qvalues = self.qtarget.forward(batch.obs_, batch.extras_)
        next_qvalues[batch.available_actions_ == 0.0] = -torch.inf
        next_qvalues: torch.Tensor = torch.max(next_qvalues, dim=-1)[0]
        next_qvalues = next_qvalues.sum(dim=-1)
        targets = batch.rewards + self.gamma * next_qvalues * (1 - batch.dones)
        return targets

    def _sample(self) -> Batch:
        return self.memory.sample(self.batch_size)

    def compute_qvalues(self, data: Batch | Observation) -> torch.Tensor:
        qvalues = super().compute_qvalues(data)
        if isinstance(data, Batch):
            qvalues = qvalues.sum(dim=-1)
        return qvalues
