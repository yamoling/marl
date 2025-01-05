from dataclasses import dataclass
from typing import Literal
from typing_extensions import Literal
from marlenv import Transition
import torch
from copy import deepcopy

from marl.models.batch import Batch
from marl.models import TransitionMemory
from marl.models.nn import CriticNN, Mixer
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
        value_network: CriticNN,
        gamma: float,
        mixer: Mixer,
        update_method: TargetParametersUpdater | Literal["soft", "hard"] = "soft",
        lr: float = 1e-4,
        batch_size: int = 64,
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
        self.mixer = mixer.to(value_network.device)
        self.target_network = deepcopy(value_network)
        self.target_mixer = deepcopy(mixer)
        self.target_network.randomize()
        self.target_mixer.randomize()
        self.update_method.add_parameters(self.network.parameters(), self.target_network.parameters())
        self.update_method.add_parameters(self.network.parameters(), self.mixer.parameters())
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.memory = TransitionMemory(5_000)
        self.device = self.network.device

    def compute(self, batch: Batch) -> torch.Tensor:
        with torch.no_grad():
            values = self.network.value(batch.states, batch.states_extras)
            values = self.mixer.forward(values, batch.states)
            next_values = self.target_network.value(batch.next_states, batch.next_states_extras)
            next_values = self.target_mixer.forward(next_values, batch.next_states)
            # Equation 2 in Haven's paper
            advantage = batch.rewards + self.gamma * next_values - values
        return advantage

    def update_step(self, transition: Transition, time_step: int) -> dict[str, float]:
        self.memory.add(transition)
        if not self.memory.can_sample(self.batch_size):
            return {}
        batch = self.memory.sample(self.batch_size).to(self.device)
        return self.update(time_step, batch)

    def update(self, time_step: int, batch: Batch) -> dict[str, float]:
        values = self.network.value(batch.states, batch.states_extras)
        with torch.no_grad():
            next_values = self.target_network.value(batch.next_states, batch.next_states_extras)
            next_values = self.mixer.forward(next_values, batch.next_states)
        targets = batch.rewards + self.gamma * next_values
        loss = torch.nn.functional.mse_loss(values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_method.update(time_step)
        return {"ir-loss": loss.item()}

    def to(self, device: torch.device):
        self.network.to(device)
        self.target_network.to(device)
        self.mixer.to(device)
        self.target_mixer.to(device)
        self.device = device
        return self

    def randomize(self, method: Literal["xavier", "orthogonal"] = "xavier"):
        self.network.randomize(method)
        self.target_network.randomize(method)
        self.mixer.randomize(method)
        self.target_mixer.randomize(method)
        return self
