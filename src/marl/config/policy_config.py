from dataclasses import dataclass
from typing import Literal, overload

from marlenv.utils import Schedule

from marl import policy
from marl.models import Policy

from .config import Config


@dataclass
class ScheduleConfig(Config[Schedule]):
    kind: Literal["constant", "linear", "exponential"]
    start: float
    end: float
    n_steps: int

    def make(self):
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
class PolicyConfig(Config[Policy]):
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
    def epsilon(kind: Literal["linear"], n_steps: int, end: float, start: float = 1.0, /) -> "EpsilonGreedyConfig": ...

    @overload
    @staticmethod
    def epsilon(kind: Literal["exponential"], end: float, n_steps: int, start: float = 1.0, /) -> "EpsilonGreedyConfig": ...

    @staticmethod
    def epsilon(*args):
        match args:
            case ("constant", value):
                schedule = ScheduleConfig.constant(value)
            case ("linear", n_steps, end):
                schedule = ScheduleConfig.linear(1.0, end, n_steps)
            case ("linear", n_steps, end, start):
                schedule = ScheduleConfig.linear(start, end, n_steps)
            case ("exponential", n_steps, end):
                schedule = ScheduleConfig.exponential(1.0, end, n_steps)
            case ("exponential", n_steps, end, start):
                schedule = ScheduleConfig.exponential(start, end, n_steps)
            case _:
                raise ValueError(f"Unknown argument combination kind: {args}")
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
