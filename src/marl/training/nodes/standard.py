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

    def randomize(self):
        self.qnetwork.randomize()
        return super().randomize()


class NextValues(Node[torch.Tensor]):
    """Compute the value of the next observation (max over qvalues)"""

    def __init__(self, qtarget: NN, batch: Node[Batch]):
        super().__init__([batch])
        self.qtarget = qtarget
        self.batch = batch

    def to(self, device: torch.device):
        self.qtarget.to(device)
        return super().to(device)

    def randomize(self):
        self.qtarget.randomize()
        return super().randomize()

    def _compute_value(self) -> torch.Tensor:
        batch = self.batch.value
        # with torch.no_grad():
        next_qvalues = forward(self.qtarget, batch.obs_, batch.extras_)
        if isinstance(batch, EpisodeBatch):  # isinstance(self.qtarget, RecurrentNN):
            # For episode batches, the batch includes the initial observation
            # in order to compute the hidden state at t=0 and use it for t=1.
            # We need to remove it when considering the next qvalues.
            next_qvalues = next_qvalues[1:]
        next_qvalues[batch.available_actions_ == 0.0] = -torch.inf
        return torch.max(next_qvalues, dim=-1)[0]


class DoubleQLearning(Node[torch.Tensor]):
    def __init__(self, qnetwork: NN, qtarget: NN, batch: Node[Batch]):
        super().__init__([batch])
        self.qnetwork = qnetwork
        self.qtarget = qtarget
        self.batch = batch

    def randomize(self):
        self.qnetwork.randomize()
        self.qtarget.randomize()
        return super().randomize()

    def to(self, device: torch.device):
        self.qnetwork.to(device)
        self.qtarget.to(device)
        return super().to(device)

    def _compute_value(self) -> torch.Tensor:
        batch = self.batch.value
        target_next_qvalues = forward(self.qtarget, batch.obs_, batch.extras_)
        # Take the indices from the target network and the values from the current network
        # instead of taking both from the target network
        current_next_qvalues = forward(self.qnetwork, batch.obs_, batch.extras_)
        if isinstance(batch, EpisodeBatch):
            # See above comment in NextQValues for an explanation on the reasons for this "if"
            target_next_qvalues = target_next_qvalues[1:]
            current_next_qvalues = current_next_qvalues[1:]
        current_next_qvalues[batch.available_actions_ == 0.0] = -torch.inf
        indices = torch.argmax(current_next_qvalues, dim=-1, keepdim=True)
        next_state_values = torch.gather(target_next_qvalues, -1, indices).squeeze(-1)
        return next_state_values


class Target(Node[torch.Tensor]):
    """Compute the target qvalues based on the next qvalues and the reward"""

    def __init__(self, gamma: float, next_qvalues: Node[torch.Tensor], batch: Node[Batch]):
        super().__init__([next_qvalues, batch])
        self.gamma = gamma
        self.next_state_value = next_qvalues
        self.batch = batch

    def _compute_value(self) -> torch.Tensor:
        """Compute the target qvalues based on the next qvalues and the reward"""
        batch = self.batch.value
        next_state_value = self.next_state_value.value
        targets = batch.rewards + self.gamma * next_state_value * (1 - batch.dones)
        return targets


class TDError(Node[torch.Tensor]):
    """Compute the TD error"""

    def __init__(self, predicted: Node[torch.Tensor], target: Node[torch.Tensor]):
        super().__init__([predicted, target])
        self.predicted = predicted
        self.target = target

    def _compute_value(self) -> torch.Tensor:
        """Compute the target qvalues based on the next qvalues and the reward"""
        qvalues = self.predicted.value
        qtargets = self.target.value.detach()
        assert qvalues.shape == qtargets.shape
        td_error = qvalues - qtargets
        return td_error


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
        squared_error = masked_error**2
        if batch.importance_sampling_weights is not None:
            assert squared_error.shape == batch.importance_sampling_weights.shape
            squared_error = squared_error * batch.importance_sampling_weights
        mean_squared_error = squared_error.sum() / batch.masks.sum()
        return mean_squared_error
