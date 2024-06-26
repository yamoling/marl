from typing import Optional
import torch
from marl.models import Batch, ReplayMemory, PrioritizedMemory, Updatable, QNetwork
from marl.models.batch import EpisodeBatch
from .node import Node, ValueNode
from .intrinsic_rewards import IR


class MemoryNode(Node[Batch]):
    def __init__(self, memory: ReplayMemory, batch_size: int, device: torch.device, individual_learners: bool):
        super().__init__([])
        self.memory = memory
        self.batch_size = batch_size
        self.device = device
        self.individual_learners = individual_learners

    def to(self, device: torch.device):
        self.device = device
        return super().to(device)

    def _compute_value(self) -> Batch:
        batch = self.memory.sample(self.batch_size).to(self.device)
        if self.individual_learners:
            batch = batch.for_individual_learners()
        return batch


class PERNode(MemoryNode, Updatable):
    def __init__(self, memory: PrioritizedMemory, batch_size: int, device: torch.device, individual_learners: bool):
        super().__init__(memory, batch_size, device, individual_learners)
        self.td_error = ValueNode(torch.tensor([]))

        # Type hinting
        self.memory: PrioritizedMemory

    def set_td_error_node(self, td_error: Node[torch.Tensor]):
        self.td_error = td_error

    def update(self, time_step: int) -> dict[str, float]:
        self.memory.update(self.td_error.value.detach())
        return {"per-alpha": self.memory.alpha.value, "per-beta": self.memory.beta.value}


class QValues(Node[torch.Tensor]):
    """
    Compute the qvalues of all actions for the given batch of observations and extras.
    """

    def __init__(self, qnetwork: QNetwork, batch: Node[Batch]):
        super().__init__([batch])
        self.qnetwork = qnetwork
        self.batch = batch

    def to(self, device: torch.device):
        self.qnetwork.to(device)
        return super().to(device)

    def _compute_value(self) -> torch.Tensor:
        batch = self.batch.value
        qvalues = self.qnetwork.batch_forward(batch.obs, batch.extras)
        return qvalues

    def randomize(self):
        self.qnetwork.randomize()
        return super().randomize()


class SelectedActionQValues(Node[torch.Tensor]):
    """Compute the qvalues of the selected actions"""

    def __init__(self, qvalues: Node[torch.Tensor], batch: Node[Batch]):
        super().__init__([qvalues, batch])
        self.qvalues = qvalues
        self.batch = batch

    def _compute_value(self) -> torch.Tensor:
        qvalues = torch.gather(self.qvalues.value, index=self.batch.value.actions, dim=-1)
        return qvalues.squeeze(dim=-1)


class NextQValues(Node[torch.Tensor]):
    """Compute the qvalues of the next observation"""

    def __init__(self, qtarget: QNetwork, batch: Node[Batch]):
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
        next_qvalues = self.qtarget.batch_forward(batch.obs_, batch.extras_)
        if isinstance(batch, EpisodeBatch):  # isinstance(self.qtarget, RecurrentNN):
            # For episode batches, the batch includes the initial observation
            # in order to compute the hidden state at t=0 and use it for t=1.
            # We need to remove it when considering the next qvalues.
            next_qvalues = next_qvalues[1:]
        next_qvalues[batch.available_actions_ == 0.0] = -torch.inf
        return next_qvalues


class NextValues(Node[torch.Tensor]):
    """Compute the value of the next observation (max over next qvalues)"""

    def __init__(self, next_qvalues: Node[torch.Tensor]):
        super().__init__([next_qvalues])
        self.next_qvalues = next_qvalues

    def _compute_value(self) -> torch.Tensor:
        return torch.max(self.next_qvalues.value, dim=-1)[0]


class DoubleQLearning(Node[torch.Tensor]):
    """Compute the value of the next observation with double q-learning"""

    def __init__(self, qnetwork: QNetwork, target_next_qvalues: Node[torch.Tensor], batch: Node[Batch]):
        super().__init__([batch])
        self.qnetwork = qnetwork
        self.target_next_qvalues = target_next_qvalues
        self.batch = batch

    def randomize(self):
        self.qnetwork.randomize()
        return super().randomize()

    def to(self, device: torch.device):
        self.qnetwork.to(device)
        return super().to(device)

    def _compute_value(self):
        batch = self.batch.value
        target_next_qvalues = self.target_next_qvalues.value
        # Take the indices from the target network and the values from the current network
        # instead of taking both from the target network
        current_next_qvalues = self.qnetwork.batch_forward(batch.obs_, batch.extras_)
        if isinstance(batch, EpisodeBatch):
            # See above comment in NextQValues for an explanation on the reasons for this "if"
            current_next_qvalues = current_next_qvalues[1:]
        current_next_qvalues[batch.available_actions_ == 0.0] = -torch.inf
        indices = torch.argmax(current_next_qvalues, dim=-1, keepdim=True)
        next_state_values = torch.gather(target_next_qvalues, -1, indices).squeeze(-1)
        return next_state_values


class Target(Node[torch.Tensor]):
    """Compute the target qvalues based on the next qvalues and the reward"""

    def __init__(self, gamma: float, next_values: Node[torch.Tensor], batch: Node[Batch]):
        super().__init__([next_values, batch])
        self.gamma = gamma
        self.next_state_value = next_values
        self.batch = batch

    def _compute_value(self) -> torch.Tensor:
        """Compute the target qvalues based on the next qvalues and the reward"""
        batch = self.batch.value
        next_state_value = self.next_state_value.value
        targets = batch.rewards + self.gamma * next_state_value * (1 - batch.dones)
        return targets


class TDError(Node[torch.Tensor]):
    """Compute the Temporal-Difference error."""

    def __init__(
        self,
        predicted: Node[torch.Tensor],
        target: Node[torch.Tensor],
        memory: PERNode | MemoryNode | IR,
    ):
        """We take the memory node as parameter in order to set the TD-error in case of PER."""
        super().__init__([predicted, target])
        self.qvalues = predicted
        self.target = target
        if isinstance(memory, IR):
            if not isinstance(memory.batch, (MemoryNode, PERNode)):
                raise ValueError("Unhandled case: batch in IR is not a MemoryNode nor a PERNode.")
            memory = memory.batch
        if isinstance(memory, PERNode):
            memory.set_td_error_node(self)

    def _compute_value(self) -> torch.Tensor:
        qvalues = self.qvalues.value
        qtargets = self.target.value.detach()
        assert qvalues.shape == qtargets.shape
        td_error = qvalues - qtargets
        return td_error


class MSELoss(Node[torch.Tensor]):
    """MSE loss node"""

    def __init__(
        self,
        td_error: Node[torch.Tensor],
        batch: Node[Batch],
    ):
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


class BackpropNode(Node[None], Updatable):
    def __init__(
        self,
        loss: Node[torch.Tensor],
        parameters: list[torch.nn.Parameter],
        optimiser: torch.optim.Optimizer,
        grad_norm_clipping: Optional[float] = None,
    ):
        super().__init__([loss])
        self.loss = loss
        self.parameters = parameters
        self.optimiser = optimiser
        self.grad_norm_clipping = grad_norm_clipping

    def _compute_value(self):
        return None

    def update(self, time_step: int) -> dict[str, float]:
        loss = self.loss.value
        logs = {"loss": loss.item()}
        self.optimiser.zero_grad()
        loss.backward()
        if self.grad_norm_clipping is not None:
            logs["grad_norm"] = torch.nn.utils.clip_grad_norm_(self.parameters, self.grad_norm_clipping).item()
        self.optimiser.step()
        return logs
