from typing import Optional
from dataclasses import dataclass
from serde import serde

from abc import abstractmethod


@serde
@dataclass(eq=False)
class Schedule:
    name: str
    start_value: float
    end_value: float

    def __init__(self, start_value: float, end_value: float):
        self.start_value = start_value
        self.end_value = end_value
        self.name = self.__class__.__name__

    def rounded(self, n_digits: int = 0) -> "RoundedSchedule":
        return RoundedSchedule(self, n_digits)

    @abstractmethod
    def update(self, step: Optional[int] = None):
        """Update the value of the schedule. Force a step if given."""

    @property
    @abstractmethod
    def value(self) -> float:
        """Returns the current value of the schedule"""

    @staticmethod
    def constant(value: float):
        return ConstantSchedule(value)

    @staticmethod
    def linear(start_value: float, end_value: float, n_steps: int):
        return LinearSchedule(start_value, end_value, n_steps)

    @staticmethod
    def exp(start_value: float, end_value: float, n_steps: int):
        return ExpSchedule(start_value, end_value, n_steps)

    # Operator overloading
    def __mul__(self, other):
        return self.value * other

    def __rmul__(self, other):
        return self.value * other

    def __pow__(self, exp: float) -> float:
        return self.value**exp

    def __rpow__(self, other):
        return other**self.value

    def __add__(self, other):
        return self.value + other

    def __neg__(self) -> float:
        return -self.value

    def __pos__(self) -> float:
        return +self.value

    def __radd__(self, other):
        return self.value + other

    def __sub__(self, other):
        return self.value - other

    def __rsub__(self, other):
        return other - self.value

    def __div__(self, other):
        return self.value / other

    def __rdiv__(self, other):
        return other / self.value

    def __truediv__(self, other):
        return self.value / other

    def __rtruediv__(self, other):
        return other / self.value

    def __lt__(self, other) -> bool:
        return self.value < other

    def __le__(self, other) -> bool:
        return self.value <= other

    def __gt__(self, other) -> bool:
        return self.value > other

    def __ge__(self, other) -> bool:
        return self.value >= other

    def __eq__(self, other) -> bool:
        return self.value == other

    def __ne__(self, other) -> bool:
        return self.value != other

    def __float__(self):
        return self.value

    def __int__(self) -> int:
        return int(self.value)


@serde
@dataclass(eq=False)
class LinearSchedule(Schedule):
    n_steps: int

    def __init__(self, start_value: float, end_value: float, n_steps: int):
        super().__init__(start_value, end_value)
        self.n_steps = n_steps
        self.current_value = self.start_value
        # y = ax + b
        self.a = (self.end_value - self.start_value) / self.n_steps
        self.b = self.start_value
        self.t = 0

    def update(self, step: Optional[int] = None):
        if step is None:
            self.t += 1
        else:
            self.t = step
        if self.t >= self.n_steps:
            self.current_value = self.end_value
        else:
            self.current_value = self.a * (self.t) + self.b

    @property
    def value(self) -> float:
        return self.current_value


@serde
@dataclass(eq=False)
class ExpSchedule(Schedule):
    """Exponential schedule. After n_steps, the value will be min_value.

    Update formula is next_value = start_value * (min_value / start_value) ** (step / (n - 1))
    """

    n_steps: int

    def __init__(self, start_value: float, min_value: float, n_steps: int):
        super().__init__(start_value, min_value)
        self.n_steps = n_steps
        self.current_value = self.start_value
        self.base = self.end_value / self.start_value
        self.last_update_step = self.n_steps - 1
        self.t = 0

    def update(self, step: Optional[int] = None):
        if step is not None:
            raise NotImplementedError("ExpSchedule does not support direct step updates")
        if self.t >= self.last_update_step:
            self.current_value = self.end_value
        else:
            self.t += 1
            self.current_value = self.start_value * (self.base) ** (self.t / (self.n_steps - 1))

    @property
    def value(self) -> float:
        return self.current_value


class ConstantSchedule(Schedule):
    def __init__(self, value: float):
        super().__init__(value, value)
        self._value = value

    def update(self, step=None):
        pass

    @property
    def value(self) -> float:
        return self._value


class RoundedSchedule(Schedule):
    def __init__(self, schedule: Schedule, n_digits: int):
        self.schedule = schedule
        self._value = schedule.value
        self.n_digits = n_digits

    def update(self, step: int | None = None):
        return self.schedule.update(step)

    @property
    def value(self) -> float:
        return round(self.schedule.value, self.n_digits)


class MultiSchedule(Schedule):
    def __init__(self, schedules: dict[int, Schedule]):
        first_schedule = schedules.get(0, None)
        if first_schedule is None:
            raise ValueError("First schedule must start at t=0")
        sorted_steps = sorted(schedules.keys())
        start = first_schedule.start_value
        end = schedules[sorted_steps[-1]].end_value
        super().__init__(start, end)

        self.schedules = [(t, schedules[t]) for t in sorted_steps]
        self.current_schedule = first_schedule
        self.t = 0

    def update(self, step: int | None = None):
        if step is None:
            self.t += 1
        else:
            self.t = step

        index = 0
        while index < len(self.schedules) - 1 and self.t >= self.schedules[index + 1][0]:
            index += 1
        t, self.current_schedule = self.schedules[index]
        self.current_schedule.update(self.t - t)

    @property
    def value(self):
        return self.current_schedule.value
