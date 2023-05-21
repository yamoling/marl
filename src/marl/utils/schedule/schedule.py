from ..summarizable import Summarizable
from abc import abstractmethod

class Schedule(Summarizable):
    def __init__(self, start_value: float, min_value: float, n_steps: int):
        self._start_value = start_value
        self._min_value = min_value
        self._n_steps = n_steps
        self._current_step = 0

    @abstractmethod
    def update(self, step: int=None):
        """Update the value of the schedule. Force a step if given."""
        
    @property
    @abstractmethod
    def value(self) -> float:
        """Returns the current value of the schedule"""
    
    @property
    def min_value(self) -> float:
        return self._min_value
    
    @property
    def n_steps(self) -> int:
        return self._n_steps
    
    def summary(self) -> dict[str, ]:
        return {
            **super().summary(),
            "start_value": self._start_value,
            "min_value": self._min_value,
            "n_steps": self._n_steps,
        }

class LinearSchedule(Schedule):
    def __init__(self, start_value: float, min_value: float, n_steps: int):
        super().__init__(start_value, min_value, n_steps)
        self._decrease = (start_value - min_value) / n_steps
        self._value = start_value

    def update(self, step=None):
        if step is None:
            self._current_step += 1
            self._value = max(self._value - self._decrease, self._min_value)
        else:
            diff = step - self._current_step
            delta = diff * self._decrease
            self._value = max(self._value - delta, self._min_value)    
            self._current_step = step
            

    @property
    def value(self) -> float:
        return self._value

    
class ExpSchedule(Schedule):
    """Exponential schedule. After n_steps, the value will be min_value.
    
    Update formula is next_value = start_value * (min_value / start_value) ** (step / (n - 1))
    """
    def __init__(self, start_value: float, min_value: float, n_steps: int):
        super().__init__(start_value, min_value, n_steps)
        self._value = start_value
        self._base = min_value / start_value
        self._step = 0
        self._last_update_step = self._n_steps - 1
        
    def update(self, step=None):
        if step is not None:
            raise NotImplementedError("ExpSchedule does not support direct step updates")
        if self._step >= self._last_update_step:
            self._value = self._min_value
        else:
            self._step += 1
            self._value = self._start_value * (self._base) ** (self._step / (self._n_steps - 1))

    @property
    def value(self) -> float:
        return self._value
    