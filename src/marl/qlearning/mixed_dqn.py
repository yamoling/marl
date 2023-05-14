import torch
from torch.optim import Optimizer
from rlenv import Observation
from marl.models import TransitionMemory, TransitionsBatch
from marl.nn import LinearNN
from marl.policy import Policy
from copy import deepcopy

from .dqn import DQN
from .mixers import Mixer


class MixedDQN(DQN):
    def __init__(
            self, 
            qnetwork: LinearNN, 
            mixer: Mixer,
            gamma=0.99, 
            tau=0.01, 
            batch_size=64, 
            lr=0.0001, 
            optimizer: Optimizer = None, 
            train_policy: Policy = None, 
            test_policy: Policy = None, 
            memory: TransitionMemory = None, 
            device: torch.device = None):
        if optimizer is None:
            optimizer = torch.optim.Adam(list(qnetwork.parameters()) + list(mixer.parameters()), lr=lr)
        super().__init__(qnetwork, gamma, tau, batch_size, lr, optimizer, train_policy, test_policy, memory, device)
        self.mixer = mixer.to(self._device, non_blocking=True)
        self.target_mixer = deepcopy(mixer).to(self._device, non_blocking=True)

    def process_batch(self, batch: TransitionsBatch) -> TransitionsBatch:
        return batch
    
    def compute_targets(self, batch: TransitionsBatch) -> torch.Tensor:
        next_qvalues = self._qtarget.forward(batch.obs_, batch.extras_)
        next_qvalues[batch.available_actions_ == 0.0] = -torch.inf
        next_qvalues: torch.Tensor = torch.max(next_qvalues, dim=-1)[0]
        next_qvalues = self.target_mixer.forward(next_qvalues, batch.states_)
        targets = batch.rewards + self.gamma * next_qvalues * (1 - batch.dones)
        return targets

    def _sample(self) -> TransitionsBatch:
        return self.memory.sample(self._batch_size)
    
    def _target_soft_update(self):
        for param, target_param in zip(self.mixer.parameters(), self.target_mixer.parameters()):
            new_value = (1-self._tau) * target_param.data + self._tau * param.data
            target_param.data.copy_(new_value, non_blocking=True)
        return super()._target_soft_update()

    def compute_qvalues(self, data: TransitionsBatch | Observation) -> torch.Tensor:
        qvalues = super().compute_qvalues(data)
        if isinstance(data, TransitionsBatch):
            qvalues = self.mixer.forward(qvalues, data.states)
        return qvalues
    
    def save(self, to_directory: str):
        super().save(to_directory)
        self.mixer.save(to_directory)
    
    def load(self, from_directory: str):
        super().load(from_directory)
        self.mixer.load(from_directory)


    def summary(self) -> dict[str, ]:
        return {
            **super().summary(),
            "mixer": self.mixer.summary()
        }

    @classmethod
    def from_summary(cls, summary: dict[str, ]):
        from marl.qlearning import mixers
        summary["mixer"] = mixers.from_summary(summary['mixer'])
        return super().from_summary(summary)
    
    