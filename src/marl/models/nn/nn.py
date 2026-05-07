import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import torch
from torch import Tensor

from marl.utils import Serializable


def randomize(init_fn: Callable[[torch.Tensor], Any], nn: torch.nn.Module):
    for param in nn.parameters():
        if len(param.data.shape) < 2:
            init_fn(param.data.view(1, -1))
        else:
            init_fn(param.data)


@dataclass
class NN(torch.nn.Module, Serializable):
    """Parent class of all neural networks"""

    output_shape: tuple[int, ...]

    def __post_init__(self):
        torch.nn.Module.__init__(self)
        Serializable.__post_init__(self)

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def output_size(self):
        return math.prod(self.output_shape)

    def randomize(self, method: Literal["xavier", "orthogonal"] | Callable[[torch.Tensor], Any] = "xavier"):
        match method:
            case "xavier":
                randomize(torch.nn.init.xavier_uniform_, self)
            case "orthogonal":
                randomize(torch.nn.init.orthogonal_, self)
            case other:
                randomize(other, self)

    def to(self, device: torch.device):  # type: ignore
        super().to(device)
        for child in self.children():
            if isinstance(child, torch.nn.ModuleList):
                for subchild in child:
                    subchild.to(device)
        return self

    @property
    def is_recurrent(self):
        for nn in self.children():
            if isinstance(nn, NN) and nn.is_recurrent:
                return True
            if isinstance(nn, (torch.nn.GRU, torch.nn.GRUCell, torch.nn.LSTM, torch.nn.LSTMCell)):
                return True
        return False

    @property
    def device(self):
        return self.parameters().__next__().device

    def __repr__(self):
        return f"{self.name} (on {self.device})"

    def weights_filename(self, directory: Path):
        return directory / self.__class__.__name__

    def save(self, directory: Path):
        state_dict = self.state_dict()
        if len(state_dict) == 0:
            return
        torch.save(state_dict, self.weights_filename(directory))

    def load(self, directory: Path):
        if len(self.state_dict()) == 0:
            return
        self.load_state_dict(torch.load(self.weights_filename(directory), weights_only=True))

    def __hash__(self):
        return id(self)


class RecurrentNN(NN):
    def __init__(self, output_shape: tuple[int, ...]):
        super().__init__(output_shape)
        self._hidden_states: Tensor | None = None
        self._saved_hidden_states: Tensor | None = None

    def reset_hidden_states(self):
        """Reset the hidden states"""
        self._hidden_states = None

    def train(self, mode: bool = True):
        if not mode:
            # Set test mode: save training hidden states
            self._saved_hidden_states = self._hidden_states
            self.reset_hidden_states()
        else:
            # Set train mode
            if not self.training:
                # if not already in train mode, restore hidden states
                self._hidden_states = self._saved_hidden_states
        return super().train(mode)

    @property
    def is_recurrent(self):
        return True

    def __hash__(self):
        return id(self)


ActivationType = Literal["relu", "tanh", "sigmoid", "leaky-relu"]


def get_activation(activation: Literal["sigmoid", "tanh", "relu", "leaky-relu"]):
    match activation:
        case "sigmoid":
            return torch.nn.Sigmoid()
        case "tanh":
            return torch.nn.Tanh()
        case "relu":
            return torch.nn.ReLU()
        case "leaky-relu":
            return torch.nn.LeakyReLU()
        case other:
            raise ValueError(f"Unsupported activation: {other}")
