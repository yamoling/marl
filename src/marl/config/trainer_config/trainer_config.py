from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Literal

from marl.utils import Serializable

from ..ir_config import IRConfig

if TYPE_CHECKING:
    from marl import Trainer


@dataclass
class TrainerConfig(Serializable):
    kind: Literal["dqn", "qmix", "qplex", "maven", "mappo", "ippo", "option-critic", "ppo-option-critic"]
    _: KW_ONLY
    gamma: float = 0.99
    ir_config: IRConfig | None = None
    grad_norm_clipping: float | None = None
    batch_size: int = 64
    train_interval: tuple[int, Literal["step", "episode"]] = (5, "step")

    def make(self) -> Trainer: ...

    @staticmethod
    def no_train():
        return
