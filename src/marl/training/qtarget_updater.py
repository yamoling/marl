from dataclasses import dataclass
from abc import abstractmethod
from typing import Iterable

import torch


@dataclass
class TargetParametersUpdater:
    name: str

    def __init__(self):
        self.name = self.__class__.__name__
        self.parameters = list[torch.nn.Parameter]()
        self.target_params = list[torch.nn.Parameter]()

    def add_parameters(self, parameters: Iterable[torch.nn.Parameter], target_params: Iterable[torch.nn.Parameter]):
        parameters = list(parameters)
        target_params = list(target_params)
        for param, target in zip(parameters, target_params):
            assert param.shape == target.shape, "Parameter and target parameter shapes must match"
        self.parameters.extend(parameters)
        self.target_params.extend(target_params)

    @abstractmethod
    def update(self, time_step: int) -> dict[str, float]:
        """Update the target network parameters based on the current network parameters and return the logs."""


@dataclass
class HardUpdate(TargetParametersUpdater):
    update_period: int

    def __init__(self, update_period: int):
        super().__init__()
        assert update_period > 0, "Update period must be positive"
        self.update_period = update_period
        self.update_num = 0

    def update(self, step_num: int) -> dict[str, float]:
        self.update_num += 1
        if self.update_num % self.update_period == 0:
            for param, target in zip(self.parameters, self.target_params):
                target.data.copy_(param.data, non_blocking=True)
        return {}


@dataclass
class SoftUpdate(TargetParametersUpdater):
    tau: float

    def __init__(self, tau: float = 0.01):
        super().__init__()
        assert 0 < tau < 1, "Soft update ratio must be between 0 and 1"
        self.tau = tau

    def update(self, time_step: int) -> dict[str, float]:
        for param, target in zip(self.parameters, self.target_params):
            new_value = (1 - self.tau) * target.data + self.tau * param.data
            target.data.copy_(new_value, non_blocking=True)
        return {}
