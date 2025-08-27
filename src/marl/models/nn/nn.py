from typing import Optional
from dataclasses import dataclass
from typing import Literal
from abc import ABC, abstractmethod
import torch


def randomize(init_fn, nn: torch.nn.Module):
    for param in nn.parameters():
        if len(param.data.shape) < 2:
            init_fn(param.data.view(1, -1))
        else:
            init_fn(param.data)


@dataclass
class NN(torch.nn.Module, ABC):
    """Parent class of all neural networks"""

    input_shape: tuple[int, ...]
    extras_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    name: str

    def __init__(self, input_shape: tuple[int, ...], extras_shape: tuple[int, ...], output_shape: tuple[int, ...]):
        torch.nn.Module.__init__(self)
        self.input_shape = input_shape
        self.extras_shape = extras_shape
        self.output_shape = output_shape
        self.name = self.__class__.__name__
        self._device = torch.device("cpu")

    @property
    def device(self):
        return self._device

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass"""

    def randomize(self, method: Literal["xavier", "orthogonal"] = "xavier"):
        match method:
            case "xavier":
                randomize(torch.nn.init.xavier_uniform_, self)
            case "orthogonal":
                randomize(torch.nn.init.orthogonal_, self)
            case _:
                raise ValueError(f"Unknown initialization method: {method}. Choose between 'xavier' and 'orthogonal'")

    def __hash__(self) -> int:
        # Required for deserialization (in torch.nn.module)
        return hash(self.name)

    def to(self, device: str | int | torch.device | Literal["auto"], dtype: Optional[torch.dtype] = None, non_blocking=True):  # type: ignore
        if device == "auto":
            from marl.utils import get_device

            self._device = get_device(device)  # type: ignore
        else:
            self._device = torch.device(device)
        return super().to(device, dtype, non_blocking)


@dataclass
class RecurrentNN(NN):
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
