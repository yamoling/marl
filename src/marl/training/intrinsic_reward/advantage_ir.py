from dataclasses import dataclass
from typing import Literal
from typing_extensions import Literal
from marlenv import Transition
import torch
from copy import deepcopy

from marl.models.batch import Batch
from marl.models import TransitionMemory
from marl.models.nn import Critic
from marl.training.qtarget_updater import TargetParametersUpdater, SoftUpdate, HardUpdate

from .ir_module import IRModule


@dataclass
class AdvantageIntrinsicReward(IRModule):
    """
    Computes an intrinsic reward that is the advantage of the action taken by the agent. Papers such as Haven use
    this approach https://arxiv.org/pdf/2110.07246.

    We compute the advantage as the difference between the reward obtained + the discounted value of the next state
    and the value of the current state:
    A(s_t, a_t) = r + \\gamma V(s_{t+1}) - V(s_t)
    """

    def __init__(
        self,
        value_network: Critic,
        gamma: float,
        update_method: TargetParametersUpdater | Literal["soft", "hard"] = "soft",
        lr: float = 1e-4,
        batch_size: int = 64,
        grad_norm_clipping: float | None = 10.0,
    ):
        match update_method:
            case "soft":
                update_method = SoftUpdate(1e-2)
            case "hard":
                update_method = HardUpdate(200)
        super().__init__()
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_method = update_method
        self.network = value_network
        self.target_network = deepcopy(value_network)
        self.target_network.randomize()
        self.update_method.add_parameters(self.network.parameters(), self.target_network.parameters())
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.memory = TransitionMemory(5_000)
        self._device = self.network.device
        self.grad_norm_clipping = grad_norm_clipping

    def compute(self, batch: Batch) -> torch.Tensor:
        with torch.no_grad():
            values = self.network.value(batch.states, batch.states_extras)
            next_values = self.target_network.value(batch.next_states, batch.next_states_extras)
            # Equation 2 in Haven's paper
            advantage = batch.rewards + self.gamma * next_values - values
        return advantage

    def update_step(self, transition: Transition, time_step: int) -> dict[str, float]:
        self.memory.add(transition)
        if not self.memory.can_sample(self.batch_size):
            return {}
        batch = self.memory.sample(self.batch_size).to(self._device)
        return self.update(time_step, batch)

    def update(self, time_step: int, batch: Batch) -> dict[str, float]:
        values = self.network.value(batch.states, batch.states_extras)
        with torch.no_grad():
            next_values = self.target_network.value(batch.next_states, batch.next_states_extras)
            next_values = next_values * (1 - batch.dones)
        targets = batch.rewards + self.gamma * next_values
        loss = torch.nn.functional.mse_loss(values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        logs = {"ir-loss": float(loss.item())}
        if self.grad_norm_clipping is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_norm_clipping)
            logs["ir-grad-norm"] = float(grad_norm.item())
        self.optimizer.step()
        logs = logs | self.update_method.update(time_step)
        return logs

    def to(self, device: torch.device):
        self.network.to(device)
        self.target_network.to(device)
        self._device = device
        return self

    def randomize(self, method: Literal["xavier", "orthogonal"] = "xavier"):
        self.network.randomize(method)
        self.target_network.randomize(method)
        return self
