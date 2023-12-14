import torch
from marl.nn import LinearNN, RecurrentNN, NN
from marl.models import Batch
from marl.models.batch import EpisodeBatch
from .node import Node


def forward(nn: NN, obs: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
    match nn:
        case LinearNN():
            return nn.forward(obs, extras)
        case RecurrentNN():
            return nn.forward(obs, extras)[0]
        case other:
            raise NotImplementedError(f"Unknown NN type: {type(other)}")


class QValues(Node[torch.Tensor]):
    def __init__(self, qnetwork: NN, batch: Node[Batch]):
        super().__init__([batch])
        self.qnetwork = qnetwork
        self.batch = batch

    def to(self, device: torch.device):
        self.qnetwork.to(device)
        return super().to(device)

    def _compute_value(self) -> torch.Tensor:
        batch = self.batch.value
        qvalues = forward(self.qnetwork, batch.obs, batch.extras)
        qvalues = torch.gather(qvalues, index=batch.actions, dim=-1)
        qvalues = qvalues.squeeze(dim=-1)
        return qvalues


class NextQValues(Node[torch.Tensor]):
    """Compute the next qvalues based on the next observations"""

    def __init__(self, qtarget: NN, batch: Node[Batch]):
        super().__init__([batch])
        self.qtarget = qtarget
        self.batch = batch

    def to(self, device: torch.device):
        self.qtarget.to(device)
        return super().to(device)

    def _compute_value(self) -> torch.Tensor:
        batch = self.batch.value
        with torch.no_grad():
            next_qvalues = forward(self.qtarget, batch.obs_, batch.extras_)
        if isinstance(batch, EpisodeBatch):  # isinstance(self.qtarget, RecurrentNN):
            # For episode batches, the batch includes the initial observation
            # in order to compute the hidden state at t=0 and use it for t=1.
            # We need to remove it when considering the next qvalues.
            next_qvalues = next_qvalues[1:]
        next_qvalues[batch.available_actions_ == 0.0] = -torch.inf
        next_qvalues = torch.max(next_qvalues, dim=-1)[0]
        return next_qvalues


class DoubleQLearning(Node[torch.Tensor]):
    def __init__(self, qnetwork: NN, qtarget: NN, batch: Node[Batch]):
        super().__init__([batch])
        self.qnetwork = qnetwork
        self.qtarget = qtarget
        self.batch = batch

    def to(self, device: torch.device):
        self.qnetwork.to(device)
        self.qtarget.to(device)
        return super().to(device)

    def _compute_value(self) -> torch.Tensor:
        batch = self.batch.value
        with torch.no_grad():
            target_next_qvalues = forward(self.qtarget, batch.obs_, batch.extras_)
            # Take the indices from the target network and the values from the current network
            # instead of taking both from the target network
            current_next_qvalues = forward(self.qnetwork, batch.obs_, batch.extras_)
        if isinstance(batch, EpisodeBatch):
            # See above comment in NextQValues for an explanation
            target_next_qvalues = target_next_qvalues[1:]
            current_next_qvalues = current_next_qvalues[1:]
        current_next_qvalues[batch.available_actions_ == 0.0] = -torch.inf
        indices = torch.argmax(current_next_qvalues, dim=-1, keepdim=True)
        next_qvalues = torch.gather(target_next_qvalues, -1, indices).squeeze(-1)
        return next_qvalues


class Target(Node[torch.Tensor]):
    """Compute the target qvalues based on the next qvalues and the reward"""

    def __init__(self, gamma: float, next_qvalues: Node[torch.Tensor], batch: Node[Batch]):
        super().__init__([next_qvalues, batch])
        self.gamma = gamma
        self.next_qvalues = next_qvalues
        self.batch = batch

    def _mark_for_update(self):
        return super()._mark_for_update()

    def _compute_value(self) -> torch.Tensor:
        """Compute the target qvalues based on the next qvalues and the reward"""
        batch = self.batch.value
        next_qvalues = self.next_qvalues.value
        targets = batch.rewards + self.gamma * next_qvalues * (1 - batch.dones)
        return targets

    @property
    def value(self):
        return super().value


class TDError(Node[torch.Tensor]):
    """Compute the TD error"""

    def __init__(self, predicted: Node[torch.Tensor], target: Node[torch.Tensor]):
        super().__init__([predicted, target])
        self.predicted = predicted
        self.target = target

    def _compute_value(self) -> torch.Tensor:
        """Compute the target qvalues based on the next qvalues and the reward"""
        predicted = self.predicted.value
        target = self.target.value
        assert predicted.shape == target.shape
        td_error = target - predicted
        return td_error

    def _mark_for_update(self):
        return super()._mark_for_update()


class MSELoss(Node[torch.Tensor]):
    """MSE loss node"""

    def __init__(self, td_error: Node[torch.Tensor], batch: Node[Batch]):
        super().__init__([td_error, batch])
        self.td_error = td_error
        self.batch = batch

    def _compute_value(self):
        """Masked Mean Squared Error"""
        batch = self.batch.value
        masked_error = self.td_error.value * batch.masks
        criterion = masked_error**2
        if batch.importance_sampling_weights is not None:
            assert criterion.shape == batch.importance_sampling_weights.shape
            criterion = criterion * batch.importance_sampling_weights
        # criterion = criterion.sum(dim=0)
        mean_squared_error = criterion.sum() / batch.masks.sum()
        return mean_squared_error
