from typing_extensions import Self
from typing import TypeVar, Generic
from ..summarizable import Summarizable
from abc import abstractmethod


T = TypeVar("T")


class Schedule(Summarizable, Generic[T]):
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
    
    @staticmethod
    def constant(value: float) -> Self:
        return ConstantSchedule(value)
    
    @staticmethod
    def linear(start_value: float, min_value: float, n_steps: int) -> Self:
        return LinearSchedule(start_value, min_value, n_steps)
    
    @staticmethod
    def exp(start_value: float, min_value: float, n_steps: int) -> Self:
        return ExpSchedule(start_value, min_value, n_steps)
    
    # Operator overloading
    def __mul__(self, other: T) -> T:
        return self.value * other
    
    def __rmul__(self, other: T) -> T:
        return self.value * other
    
    def __pow__(self, exp: float) -> float:
        return self.value ** exp
    
    def __add__(self, other: T) -> T:
        return self.value + other
    
    def __radd__(self, other: T) -> T:
        return self.value + other
    
    def __sub__(self, other: T) -> T:
        return self.value - other
    
    def __rsub__(self, other: T) -> T:
        return other - self.value
    
    def __div__(self, other: T) -> T:
        return self.value / other
    
    def __rdiv__(self, other: T) -> T:
        return other / self.value
    
    def __truediv__(self, other: T) -> T:
        return self.value / other
    
    def __rtruediv__(self, other: T) -> T:
        return other / self.value
    
    def __lt__(self, other: float) -> bool:
        return self.value < other
    
    def __le__(self, other: float) -> bool:
        return self.value <= other
    
    def __gt__(self, other: float) -> bool:
        return self.value > other
    
    def __ge__(self, other: float) -> bool:
        return self.value >= other
    
    def __eq__(self, other: float) -> bool:
        return self.value == other
    
    def __ne__(self, other: float) -> bool:
        return self.value != other
    
    def __float__(self) -> float:
        return self.value
    
    def __int__(self) -> int:
        return int(self.value)
    

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
    

    
class ConstantSchedule(Schedule):
    def __init__(self, value: float):
        super().__init__(value, value, 0)
        self._value = value
        
    def update(self, step=None):
        pass
    
    def summary(self) -> dict[str, ]:
        return {
            "name": self.name,
            "value": self._value,
        }

    @property
    def value(self) -> float:
        return self._value
