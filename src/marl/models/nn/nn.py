from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import torch
from torch import Tensor


def randomize(init_fn: Callable[[torch.Tensor], Any], nn: torch.nn.Module):
    for param in nn.parameters():
        if len(param.data.shape) < 2:
            init_fn(param.data.view(1, -1))
        else:
            init_fn(param.data)


def to[T: torch.nn.Module](module: T, device: torch.device) -> T:
    torch.nn.Module.to(module, device, non_blocking=True)
    for child in module.children():
        if isinstance(child, torch.nn.ModuleList):
            for subchild in child:
                d = subchild.parameters().__next__().device
                print(f"\tChild {subchild} was on {d}")
                to(subchild, device)
                d = subchild.parameters().__next__().device
                print(f"\tChild {subchild} is now on {d}")
    return module


@dataclass
class NN(torch.nn.Module):
    """Parent class of all neural networks"""

    name: str = field(init=False)

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__

    def randomize(self, method: Literal["xavier", "orthogonal"] | Callable[[torch.Tensor], Any] = "xavier"):
        match method:
            case "xavier":
                randomize(torch.nn.init.xavier_uniform_, self)
            case "orthogonal":
                randomize(torch.nn.init.orthogonal_, self)
            case other:
                randomize(other, self)

    def __hash__(self) -> int:
        # Required for deserialization (in torch.nn.module)
        return hash(self.name)

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


@dataclass(repr=False)
class RecurrentNN(NN):
    def __init__(self):
        super().__init__()
        self.hidden_states: Tensor | None = None
        self.saved_hidden_states: Tensor | None = None

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

    @property
    def is_recurrent(self):
        return True
