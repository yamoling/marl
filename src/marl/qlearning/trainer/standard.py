import torch
from marl.nn import LinearNN
from marl.models import Batch
from .node import Node

class QValuesNode(Node[torch.Tensor]):
    def __init__(self, nn: LinearNN, batch: Node[Batch]):
        super().__init__()
        self.batch = batch
        self.nn = nn

    @property
    def value(self) -> torch.Tensor:
        batch = self.batch.value
        qvalues = self.nn.forward(batch.obs, batch.extras)
        qvalues[batch.available_actions == 0.0] = -torch.inf
        qvalues = torch.gather(qvalues, index=batch.actions, dim=-1)
        qvalues = qvalues.squeeze(dim=-1)
        return qvalues


class NextQValuesNode(Node[torch.Tensor]):
    """Compute the next qvalues based on the next observations"""
    def __init__(self, qtarget: LinearNN, batch: Node[Batch]):
        super().__init__()
        self.batch = batch
        self.qtarget = qtarget

    @property
    def value(self) -> torch.Tensor:
        batch = self.batch.value
        next_qvalues = self.qtarget.forward(batch.obs_, batch.extras_)
        next_qvalues[batch.available_actions_ == 0.0] = -torch.inf
        next_qvalues = torch.max(next_qvalues, dim=-1)[0]
        return next_qvalues


class DoubleQLearningNode(Node[torch.Tensor]):
    def __init__(self, qnetwork: LinearNN, qtarget: LinearNN, batch: Node[Batch]):
        super().__init__()
        self.batch = batch
        self.qnetwork = qnetwork
        self.qtarget = qtarget

    @property
    def value(self) -> torch.Tensor:
        batch = self.batch.value
        target_next_qvalues = self.qtarget.forward(batch.obs_, batch.extras_)
        # Take the indices from the target network and the values from the current network
        # instead of taking both from the target network
        current_next_qvalues = self.qnetwork.forward(batch.obs_, batch.extras_)
        current_next_qvalues[batch.available_actions_ == 0.0] = -torch.inf
        indices = torch.argmax(current_next_qvalues, dim=-1, keepdim=True)
        next_qvalues = torch.gather(target_next_qvalues, -1, indices).squeeze(-1)
        return next_qvalues


class TargetNode(Node[torch.Tensor]):
    """Compute the target qvalues based on the next qvalues and the reward"""

    def __init__(self, gamma: float, next_qvalues: Node[torch.Tensor], batch: Node[Batch]):
        super().__init__()
        self.gamma = gamma
        self.next_qvalues = next_qvalues
        self.batch = batch

    @property
    def value(self) -> torch.Tensor:
        """Compute the target qvalues based on the next qvalues and the reward"""
        batch = self.batch.value
        with torch.no_grad():
            next_qvalues = self.next_qvalues.value
            targets = batch.rewards + self.gamma * next_qvalues * (1 - batch.dones)
            return targets



class LossNode(Node[torch.Tensor]):
    """MSE loss node"""
    def __init__(self, predicted: Node[torch.Tensor], target: Node[torch.Tensor], batch: Node[Batch]):
        super().__init__()
        self.predicted = predicted
        self.target = target
        self.batch = batch

    @property
    def value(self) -> torch.Tensor:
        qvalues = self.predicted.value
        qtargets = self.target.value
        mse = (qvalues - qtargets) ** 2
        # Apply importance sampling weights if necessary
        importance_sampling_weights = self.batch.value.importance_sampling_weights
        if importance_sampling_weights is not None:
            mse = mse * importance_sampling_weights
        return torch.mean(mse)