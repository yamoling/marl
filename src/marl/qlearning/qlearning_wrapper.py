import torch
from marl.models import Batch
from marl import RLAlgoWrapper
from .qlearning import IDeepQLearning


class DeepQWrapper(RLAlgoWrapper, IDeepQLearning):
    def __init__(self, wrapped: IDeepQLearning) -> None:
        RLAlgoWrapper.__init__(self, wrapped)
        IDeepQLearning.__init__(self)
        self.algo: IDeepQLearning = self.algo

    def compute_qvalues(self, data):
        return self.algo.compute_qvalues(data)

    def compute_loss(self, qvalues: torch.Tensor, qtargets: torch.Tensor, batch: Batch) -> torch.Tensor:
        return self.algo.compute_loss(qvalues, qtargets, batch)

    def compute_targets(self, batch: Batch) -> torch.Tensor:
        return self.algo.compute_targets(batch)

    def process_batch(self, batch: Batch) -> Batch:
        return self.algo.process_batch(batch)
