from copy import deepcopy
from dataclasses import KW_ONLY, dataclass, field

import torch
from marlenv import Transition

from marl.models import TransitionMemory
from marl.models.batch import Batch
from marl.models.nn import Critic, IRModule
from marl.training.qtarget_updater import SoftUpdate, TargetParametersUpdater


@dataclass
class AdvantageIntrinsicReward(IRModule):
    """
    Computes an intrinsic reward that is the advantage of the action taken by the agent. Papers such as Haven use
    this approach https://arxiv.org/pdf/2110.07246.

    We compute the advantage as the difference between the reward obtained + the discounted value of the next state
    and the value of the current state:
    A(s_t, a_t) = r + \\gamma V(s_{t+1}) - V(s_t)
    """

    network: Critic
    gamma: float
    _: KW_ONLY
    update_method: TargetParametersUpdater = field(default_factory=lambda: SoftUpdate(0.01))
    lr: float = 1e-4
    batch_size: int = 64
    grad_norm_clipping: float | None = 10.0

    def __post_init__(self):
        super().__post_init__()
        self.target_network = deepcopy(self.network)
        self.target_network.randomize()
        self.update_method.add_parameters(self.network.parameters(), self.target_network.parameters())
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.memory = TransitionMemory(5_000)

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
        return self.update(batch, time_step)

    def update(self, batch: Batch, time_step: int) -> dict[str, float]:
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
