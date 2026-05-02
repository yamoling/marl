from dataclasses import KW_ONLY, dataclass
from typing import Literal

from marl.utils import Serializable


@dataclass
class TargetUpdaterConfig(Serializable):
    kind: Literal["soft", "hard"]
    _: KW_ONLY
    tau: float | None = None
    update_interval: int | None = None

    def make(self):
        from marl.training.qtarget_updater import HardUpdate, SoftUpdate

        match self.kind:
            case "hard":
                assert self.update_interval is not None, "update_interval must be provided for hard update"
                return HardUpdate(self.update_interval)
            case "soft":
                assert self.tau is not None, "tau must be provided for soft update"
                return SoftUpdate(self.tau)
            case _:
                raise NotImplementedError()

    @staticmethod
    def soft(tau: float):
        return TargetUpdaterConfig(kind="soft", tau=tau)

    @staticmethod
    def hard(interval: int):
        return TargetUpdaterConfig(kind="hard", update_interval=interval)

    @staticmethod
    def default():
        return TargetUpdaterConfig(kind="soft", tau=0.01)
