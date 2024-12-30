from dataclasses import dataclass

import torch

from marl.models.batch.batch import Batch
from marl.models.nn import CriticNN

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

    def __init__(self, value_network: CriticNN, gamma: float):
        super().__init__()
        self.network = value_network
        self.gamma = gamma

    def compute(self, batch: Batch) -> torch.Tensor:
        # Equation 2 in Haven's paper
        with torch.no_grad():
            values = self.network.value(batch.states, batch.states_extras)
            next_values = self.network.value(batch.next_states, batch.next_states_extras)
            advantage = batch.rewards + self.gamma * next_values - values
        return advantage

    def update(self, time_step: int) -> dict[str, float]:
        # TODO: il faut vérifier dans le papier comment le value network est entraîné
        return {}
