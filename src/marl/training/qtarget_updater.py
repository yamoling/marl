from abc import abstractmethod
from dataclasses import dataclass
from typing import Iterable

import torch

from marl.utils import Serializable


@dataclass
class TargetParametersUpdater(Serializable):
    def __post_init__(self):
        self._parameters = list[torch.nn.Parameter]()
        self._target_params = list[torch.nn.Parameter]()

    def add_parameters(self, parameters: Iterable[torch.nn.Parameter], target_params: Iterable[torch.nn.Parameter]):
        parameters = list(parameters)
        target_params = list(target_params)
        for param, target in zip(parameters, target_params):
            assert param.shape == target.shape, "Parameter and target parameter shapes must match"
        self._parameters.extend(parameters)
        self._target_params.extend(target_params)

    @abstractmethod
    def update(self, time_step: int) -> dict[str, float]:
        """Update the target network parameters based on the current network parameters and return the logs."""

    @property
    def parameters(self):
        return self._parameters

    @property
    def target_parameters(self):
        return self._target_params


@dataclass
class HardUpdate(TargetParametersUpdater):
    update_period: int = 200

    def __post_init__(self):
        super().__post_init__()
        assert self.update_period > 0, "Update period must be positive"
        self._update_num = 0

    def update(self, time_step: int) -> dict[str, float]:
        self._update_num += 1
        if self._update_num % self.update_period == 0:
            for param, target in zip(self._parameters, self._target_params):
                target.data.copy_(param.data, non_blocking=True)
        return {}


@dataclass
class SoftUpdate(TargetParametersUpdater):
    tau: float = 0.01

    def __post_init__(self):
        super().__post_init__()
        assert 0 < self.tau < 1, "Soft update ratio must be between 0 and 1"

    def update(self, time_step: int) -> dict[str, float]:
        for param, target in zip(self._parameters, self._target_params):
            new_value = (1 - self.tau) * target.data + self.tau * param.data
            target.data.copy_(new_value, non_blocking=True)
        return {}
