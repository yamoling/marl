import torch
from typing import Callable


class AbsLayer(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.abs(x)


class ReshapeLayer(torch.nn.Module):
    def __init__(self, *output_shape: int):
        super().__init__()
        self._output_shape = output_shape

    def forward(self, x: torch.Tensor):
        return x.reshape(*self._output_shape)


class LambdaLayer(torch.nn.Module):
    def __init__(self, function: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)


class BMMLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.Tensor([0])
        self.biases = torch.Tensor([0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.bmm(x, self.weights) + self.biases
