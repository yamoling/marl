from dataclasses import dataclass
from typing import Literal, overload

from marlenv.utils import Schedule

from marl import policy
from marl.models import Policy
from marl.utils import Serializable


@dataclass
class ScheduleConfig(Serializable):
    kind: Literal["constant", "linear", "exponential"]
    start: float
    end: float
    n_steps: int

    def make(self) -> Schedule:
        match self.kind:
            case "constant":
                return Schedule.constant(self.start)
            case "linear":
                return Schedule.linear(self.start, self.end, self.n_steps)
            case "exponential":
                return Schedule.exp(self.start, self.end, self.n_steps)
            case _:
                raise ValueError(f"Unknown schedule kind: {self.kind}")

    @staticmethod
    def constant(value: float):
        return ScheduleConfig("constant", value, value, 1)

    @staticmethod
    def linear(start: float, end: float, n_steps: int):
        return ScheduleConfig("linear", start, end, n_steps)

    @staticmethod
    def exponential(start: float, end: float, n_steps: int):
        return ScheduleConfig("exponential", start, end, n_steps)


@dataclass
class PolicyConfig(Serializable[Policy]):
    @staticmethod
    def argmax():
        return ArgmaxConfig()

    @staticmethod
    def softmax(n_actions: int, tau: float = 1.0):
        return SoftmaxConfig(n_actions, tau)

    @overload
    @staticmethod
    def epsilon(kind: Literal["constant"], value: float, /) -> "EpsilonGreedyConfig": ...

    @overload
    @staticmethod
    def epsilon(kind: Literal["linear"], start: float, end: float, n_steps: int, /) -> "EpsilonGreedyConfig": ...

    @overload
    @staticmethod
    def epsilon(kind: Literal["exponential"], start: float, end: float, n_steps: int, /) -> "EpsilonGreedyConfig": ...

    @staticmethod
    def epsilon(kind: Literal["constant", "linear", "exponential"], /, *args):
        match kind:
            case "constant":
                schedule = ScheduleConfig.constant(*args)
            case "linear":
                schedule = ScheduleConfig.linear(*args)
            case "exponential":
                schedule = ScheduleConfig.exponential(*args)
            case _:
                raise ValueError(f"Unknown epsilon kind: {kind}")
        return EpsilonGreedyConfig(schedule)


@dataclass
class EpsilonGreedyConfig(PolicyConfig):
    schedule: ScheduleConfig

    def make(self):
        return policy.EpsilonGreedy(self.schedule.make())


@dataclass
class SoftmaxConfig(PolicyConfig):
    n_actions: int
    tau: float = 1.0

    def make(self):
        from marl.policy import SoftmaxPolicy

        return SoftmaxPolicy(self.n_actions, self.tau)


class ArgmaxConfig(PolicyConfig):
    def make(self):
        return policy.ArgMax()
