import torch
from rlenv import Observation
from marl.models import Batch, ReplayMemory
from marl.policy import Policy
from marl.qlearning import IDeepQLearning

from .algo_wrapper import AlgoWrapper


class DeepQWrapper(AlgoWrapper, IDeepQLearning):
    def __init__(self, algo: IDeepQLearning) -> None:
        AlgoWrapper.__init__(self, algo)
        IDeepQLearning.__init__(self)
        # Type hinting
        self.algo: IDeepQLearning

    @property
    def gamma(self) -> float:
        return self.algo.gamma
    
    @property
    def memory(self) -> ReplayMemory:
        return self.algo.memory
    
    @property
    def policy(self) -> Policy:
        return self.algo.policy

    def compute_qvalues(self, data: Batch | Observation):
        return self.algo.compute_qvalues(data)

    def compute_loss(self, qvalues: torch.Tensor, qtargets: torch.Tensor, batch: Batch) -> torch.Tensor:
        return self.algo.compute_loss(qvalues, qtargets, batch)

    def compute_targets(self, batch: Batch) -> torch.Tensor:
        return self.algo.compute_targets(batch)

    def process_batch(self, batch: Batch) -> Batch:
        return self.algo.process_batch(batch)
    