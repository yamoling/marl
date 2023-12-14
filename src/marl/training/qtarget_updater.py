from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch


@dataclass
class TargetParametersUpdater(ABC):
    name: str

    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def update(self, current_params: list[torch.nn.Parameter], target_params: list[torch.nn.Parameter]) -> None:
        """Update the target network parameters based on the current network parameters"""


@dataclass
class HardUpdate(TargetParametersUpdater):
    update_period: int

    def __init__(self, update_period: int):
        super().__init__()
        assert update_period > 0, "Update period must be positive"
        self.update_period = update_period
        self.update_num = 0

    def update(self, current_params: list[torch.nn.Parameter], target_params: list[torch.nn.Parameter]):
        self.update_num += 1
        if self.update_num % self.update_period == 0:
            for param, target in zip(current_params, target_params):
                target.data.copy_(param.data, non_blocking=True)


@dataclass
class SoftUpdate(TargetParametersUpdater):
    tau: float

    def __init__(self, tau: float):
        super().__init__()
        assert 0 < tau < 1, "Soft update ratio must be between 0 and 1"
        self.tau = tau

    def update(self, current_params: list[torch.nn.Parameter], target_params: list[torch.nn.Parameter]):
        for param, target in zip(current_params, target_params):
            new_value = (1 - self.tau) * target.data + self.tau * param.data
            target.data.copy_(new_value, non_blocking=True)
