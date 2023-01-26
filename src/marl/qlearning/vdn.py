import torch
from rlenv import Observation
from marl.models import Batch
from .rdqn import RDQN


class VDN(RDQN):
    """
    Value Decomposition Network is a Q-learning extension that applies to cooperative multi-agent environments
    with a discrete action space.
    VDN optimises the sum of rewards instead of the individual rewards of each agent.
    """
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

    def compute_loss(self, qvalues: torch.Tensor, qtargets: torch.Tensor, batch: Batch) -> torch.Tensor:
        l = super().compute_loss(qvalues, qtargets, batch)
        return l

    def summary(self) -> dict[str,]:
        summary = super().summary()
        summary["name"] = "Value Decomposition Network"
        return summary