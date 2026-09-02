from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Iterable

import torch

from marl.utils import Serializable
from marl.utils.tuning import tuning


@dataclass
class TargetParametersUpdater(Serializable):
    def __post_init__(self):
        self._parameters = list[torch.nn.Parameter]()
        self._target_params = list[torch.nn.Parameter]()

    def add_parameters(self, parameters: Iterable[torch.nn.Parameter], target_params: Iterable[torch.nn.Parameter]):
        for param, target in zip(parameters, target_params):
            assert param.shape == target.shape, "Parameter and target parameter shapes must match"
            self._parameters.append(param)
            self._target_params.append(target)

    @abstractmethod
    def update(self, time_step: int) -> dict[str, float]:
        """Update the target network parameters based on the current network parameters and return the logs."""

    @property
    def parameters(self):
        return self._parameters

    @property
    def target_parameters(self):
        return self._target_params

    def __hash__(self):
        return hash(id(self))


@dataclass
class HardUpdate(TargetParametersUpdater):
    update_period: int = field(default=200, metadata=tuning(50, 2000))

    def __post_init__(self):
        super().__post_init__()
        assert self.update_period > 0, "Update period must be positive"
        self._update_num = 0

    @torch.no_grad()
    def update(self, time_step: int) -> dict[str, float]:
        """
        Copy the current parameters into the target parameters every `update_period` calls.

        Uses `torch._foreach_copy_` to copy all parameter tensors in a single fused op instead of
        looping in Python, falling back to a per-tensor `copy_` if the foreach variant is unavailable.

        @ai-generated
        """
        self._update_num += 1
        if self._update_num % self.update_period == 0:
            try:
                torch._foreach_copy_(self._target_params, self._parameters)
            except (RuntimeError, NotImplementedError):
                for param, target in zip(self._parameters, self._target_params):
                    target.data.copy_(param.data, non_blocking=True)
        return {}

    def __hash__(self):
        return hash(id(self))


@dataclass
class SoftUpdate(TargetParametersUpdater):
    tau: float = field(default=0.01, metadata=tuning(1e-4, 1e-1, log=True))

    def __post_init__(self):
        super().__post_init__()
        assert 0 < self.tau < 1, "Soft update ratio must be between 0 and 1"

    @torch.no_grad()
    def update(self, time_step: int) -> dict[str, float]:
        """
        Interpolate the target parameters towards the current parameters by a factor of `tau`.

        Uses a single fused `torch._foreach_lerp_` call, equivalent to
        `(1 - tau) * target + tau * param` for every tensor, instead of allocating temporaries and
        looping in Python per parameter. Falls back to a per-tensor `lerp_` loop if the foreach
        variant is unavailable or fails for the given device/dtype combination.

        @ai-generated
        """
        try:
            torch._foreach_lerp_(self._target_params, self._parameters, self.tau)
        except (RuntimeError, NotImplementedError):
            for param, target in zip(self._parameters, self._target_params):
                target.data.lerp_(param.data, self.tau)
        return {}

    def __hash__(self):
        return hash(id(self))
