from typing import Optional
from marlenv import Observation
from abc import ABC, abstractmethod
import torch

from .qnetwork import QNetwork


class MAIC(ABC):
    @abstractmethod
    def get_values_and_comms(
        self, obs: torch.Tensor, extras: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the Q-values and return Q-values and Computed messages"""


class MAICNN(QNetwork):
    def __init__(self, input_shape: tuple[int, ...], extras_shape: tuple[int, ...], output_shape: tuple[int, ...]):
        super().__init__(input_shape, extras_shape, output_shape)
        self.hidden_states: Optional[torch.Tensor] = None
        self.saved_hidden_states = None

    def reset_hidden_states(self):
        """Reset the hidden states"""
        self.hidden_states = None

    def train(self, mode: bool = True):
        if not mode:
            # Set test mode: save training hidden states
            self.saved_hidden_states = self.hidden_states
            self.reset_hidden_states()
        else:
            # Set train mode
            if not self.training:
                # if not already in train mode, restore hidden states
                self.hidden_states = self.saved_hidden_states
        return super().train(mode)

    def to_tensor(self, obs: Observation) -> tuple[torch.Tensor, torch.Tensor]:
        extras = torch.from_numpy(obs.extras).unsqueeze(0).to(self.device)
        obs_tensor = torch.from_numpy(obs.data).unsqueeze(0).to(self.device)
        return obs_tensor, extras

    def qvalues(self, obs: Observation):
        """Compute the Q-values"""
        obs_tensor, extra_tensor = self.to_tensor(obs)
        objective_qvalues, _ = self.forward(obs_tensor, extra_tensor)
        objective_qvalues = objective_qvalues.squeeze(0)
        return objective_qvalues

    def value(self, obs: Observation) -> torch.Tensor:
        """Compute the value function. Does not update the hidden states."""
        hidden_states = self.hidden_states
        objective_qvalues = self.qvalues(obs)
        qvalues = torch.sum(objective_qvalues, dim=-1)
        agent_values = torch.max(qvalues, dim=-1).values
        self.hidden_states = hidden_states
        return agent_values.mean(dim=-1)

    @abstractmethod
    def forward(self, obs: torch.Tensor, extras: torch.Tensor) -> ...:
        """Compute the Q-values"""

    def set_testing(self, test_mode: bool = True):
        """Set the network in testing mode"""
        self.test_mode = test_mode

    def batch_forward(self, obs: torch.Tensor, extras: torch.Tensor):
        """
        Compute the Q-values for a batch of observations (multiple episodes) during training.

        In this case, the RNN considers hidden states=None.
        """
        self.test_mode = False
        saved_hidden_states = self.hidden_states
        self.reset_hidden_states()
        logs = []
        losses = []
        q_values, returns_ = self.forward(obs, extras)
        if "logs" in returns_:
            logs.append(returns_["logs"])
            del returns_["logs"]
        losses.append(returns_)
        self.hidden_states = saved_hidden_states
        return q_values, logs, losses
        # self.test_mode = False
        # bs = obs.shape[1]
        # self.reset_hidden_states(bs)
        # qvalues = []
        # logs = []
        # losses = []
        # for t in range(len(obs)):  # case of Episode Batch
        #     current_q_values, returns_ = self.forward(obs[t], extras[t])
        #     qvalues.append(current_q_values)
        #     if "logs" in returns_:
        #         logs.append(returns_["logs"])
        #         del returns_["logs"]
        #     losses.append(returns_)

        # qvalues = torch.stack(qvalues, dim=0)

        # return qvalues, logs, losses
