from typing_extensions import Self
from typing import Any, TypeVar, Generic, Optional
from dataclasses import dataclass

from abc import abstractmethod


T = TypeVar("T")


@dataclass
class Schedule(Generic[T]):
    name: str
    start_value: float
    min_value: float
    n_steps: int


    def __init__(self, start_value: float, min_value: float, n_steps: int):
        self.start_value = start_value
        self.min_value = min_value
        self.n_steps = n_steps
        self.name = self.__class__.__name__
        self.current_step = 0

    @abstractmethod
    def update(self, step: Optional[int]=None):
        """Update the value of the schedule. Force a step if given."""

    @property
    @abstractmethod
    def value(self) -> float:
        """Returns the current value of the schedule"""

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

    def __pow__(self, exp: float) -> T:
        return self.value**exp

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
        self.decrease = (self.start_value - self.min_value) / self.n_steps
        self.current_value = self.start_value

    def update(self, step: Optional[int]=None):
        if step is None:
            self.current_step += 1
            self.current_value = max(self.value - self.decrease, self.min_value)
        else:
            diff = step - self.current_step
            delta = diff * self.decrease
            self.current_value = max(self.value - delta, self.min_value)
            self.current_step = step

    @property
    def value(self) -> float:
        return self.current_value


class ExpSchedule(Schedule):
    """Exponential schedule. After n_steps, the value will be min_value.

    Update formula is next_value = start_value * (min_value / start_value) ** (step / (n - 1))
    """

    def __init__(self, start_value: float, min_value: float, n_steps: int):
        super().__init__(start_value, min_value, n_steps)
        self.current_value = self.start_value
        self.base = self.min_value / self.start_value
        self.last_update_step = self.n_steps - 1

    def update(self, step: Optional[int]=None):
        if step is not None:
            raise NotImplementedError("ExpSchedule does not support direct step updates")
        if self.current_step >= self.last_update_step:
            self.current_value = self.min_value
        else:
            self.current_step += 1
            self.current_value = self.start_value * (self.base) ** (self.current_step / (self.n_steps - 1))

    @property
    def value(self) -> float:
        return self.current_value


class ConstantSchedule(Schedule):
    def __init__(self, value: float):
        super().__init__(value, value, 0)
        self._value = value

    def update(self, step=None):
        pass

    @property
    def value(self) -> float:
        return self._value
