from abc import abstractmethod
from dataclasses import KW_ONLY, dataclass, field

from marl.utils.tuning import tuning

from .serialization import Serializable


@dataclass
class Schedule(Serializable):
    """
    Schedules the value of a varaible over time.
    """

    _: KW_ONLY
    start_value: float = field(metadata=tuning(1e-2, 1.0))
    end_value: float = field(metadata=tuning(1e-2, 1.0, log=True))
    n_steps: int = field(metadata=tuning(0, 1_000_000, step=5000))

    def __post_init__(self):
        super().__post_init__()
        self._t = 0
        self._current_value = self.start_value

    def update(self, step: int | None = None):
        """Update the value of the schedule. Force a step if given."""
        if step is not None:
            self._t = step
        else:
            self._t += 1
        if self._t >= self.n_steps:
            self._current_value = self.end_value
        else:
            self._current_value = self._compute()

    @abstractmethod
    def _compute(self) -> float:
        """Compute the value of the schedule"""

    @property
    def value(self) -> float:
        """Returns the current value of the schedule"""
        return self._current_value

    @staticmethod
    def constant(value: float):
        return ConstantSchedule(start_value=value)

    @staticmethod
    def linear(start_value: float, end_value: float, n_steps: int):
        return LinearSchedule(start_value=start_value, end_value=end_value, n_steps=n_steps)

    @staticmethod
    def exp(start_value: float, end_value: float, n_steps: int):
        return ExpSchedule(start_value=start_value, end_value=end_value, n_steps=n_steps)

    def rounded(self, n_digits: int = 0) -> "RoundedSchedule":
        return RoundedSchedule(self, n_digits=n_digits)

    # Operator overloading
    def __mul__[T](self, other: T) -> T:
        return self.value * other  # type: ignore

    def __rmul__[T](self, other: T) -> T:
        return self.value * other  # type: ignore

    def __pow__[T](self, exp: float) -> float:
        return self.value**exp

    def __rpow__[T](self, other: T) -> T:
        return other**self.value  # type: ignore

    def __add__[T](self, other: T) -> T:
        return self.value + other  # type: ignore

    def __radd__[T](self, other: T) -> T:
        return self.value + other  # type: ignore

    def __neg__(self):
        return -self.value

    def __pos__(self):
        return +self.value

    def __sub__[T](self, other: T) -> T:
        return self.value - other  # type: ignore

    def __rsub__[T](self, other: T) -> T:
        return other - self.value  # type: ignore

    def __div__[T](self, other: T) -> T:
        return self.value // other  # type: ignore

    def __rdiv__[T](self, other: T) -> T:
        return other // self.value  # type: ignore

    def __truediv__[T](self, other: T) -> T:
        return self.value / other  # type: ignore

    def __rtruediv__[T](self, other: T) -> T:
        return other / self.value  # type: ignore

    def __lt__(self, other) -> bool:
        return self.value < other

    def __le__(self, other) -> bool:
        return self.value <= other

    def __gt__(self, other) -> bool:
        return self.value > other

    def __ge__(self, other) -> bool:
        return self.value >= other

    def __eq__(self, other) -> bool:
        if isinstance(other, Schedule):
            if self.start_value != other.start_value:
                return False
            if self.end_value != other.end_value:
                return False
            if self.n_steps != other.n_steps:
                return False
            if type(self) is not type(other):
                return False
        return self.value == other

    def __ne__(self, other) -> bool:
        return not (self.__eq__(other))

    def __float__(self):
        return self.value

    def __int__(self) -> int:
        return int(self.value)


@dataclass(eq=False)
class LinearSchedule(Schedule):
    def __post_init__(self):
        super().__post_init__()
        # y = ax + b
        self.a = (self.end_value - self.start_value) / self.n_steps
        self.b = self.start_value

    def _compute(self):
        return self.a * (self._t) + self.b

    @property
    def value(self) -> float:
        return self._current_value


@dataclass(eq=False)
class ExpSchedule(Schedule):
    """Exponential schedule. After n_steps, the value will be min_value.

    Update formula is next_value = start_value * (min_value / start_value) ** (step / (n - 1))
    """

    def __post_init__(self):
        super().__post_init__()
        self.base = self.end_value / self.start_value
        self.last_update_step = self.n_steps - 1

    def _compute(self):
        return self.start_value * (self.base) ** (self._t / (self.n_steps - 1))

    @property
    def value(self) -> float:
        return self._current_value


@dataclass(eq=False)
class ConstantSchedule(Schedule):
    end_value: float = field(init=False)
    n_steps: int = 0

    def __post_init__(self):
        super().__post_init__()
        self.end_value = self.start_value

    def update(self, step=None):
        return

    @property
    def value(self) -> float:
        return self.start_value


@dataclass(eq=False)
class RoundedSchedule(Schedule):
    schedule: Schedule
    _: KW_ONLY
    n_digits: int = 0
    start_value: float = field(init=False)
    end_value: float = field(init=False)
    n_steps: int = field(init=False)

    def __post_init__(self):
        self.start_value = round(self.schedule.start_value, self.n_digits)
        self.end_value = round(self.schedule.end_value, self.n_digits)
        self.n_steps = self.schedule.n_steps
        super().__post_init__()

    def update(self, step: int | None = None):
        return self.schedule.update(step)

    def _compute(self) -> float:
        return self.schedule._compute()

    @property
    def name(self):
        return f"Rounded{self.schedule.name}"

    @property
    def value(self) -> float:
        return round(self.schedule.value, self.n_digits)
