import math
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


class NoisyLinear(torch.nn.Module):
    """
    Noisy Linear Layer used for NoisyDQN
    Paper: https://arxiv.org/pdf/1706.10295 (Noisy Networks for Exploration)
    Code grabbed from: https://github.com/Lizhi-sjtu/DRL-code-pytorch/blob/main/3.Rainbow_DQN/network.py
    """

    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = torch.nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = torch.nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.FloatTensor(out_features, in_features))

        self.bias_mu = torch.nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = torch.nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer("bias_epsilon", torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            self.reset_noise()
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)  # type: ignore
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)  # type: ignore

        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return torch.nn.functional.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))  # 这里要除以out_features

    def reset_noise(self):
        epsilon_i = self.scale_noise(self.in_features)
        epsilon_j = self.scale_noise(self.out_features)
        self.weight_epsilon.copy_(torch.outer(epsilon_j, epsilon_i))  # type: ignore
        self.bias_epsilon.copy_(epsilon_j)  # type: ignore

    def scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x
