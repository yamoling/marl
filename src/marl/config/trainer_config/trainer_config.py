from dataclasses import KW_ONLY, dataclass
from typing import Literal

from marl.models.trainer import Trainer

from ..config import Config
from ..ir_config import IRConfig


@dataclass
class TrainerConfig[T: Trainer](Config[T]):
    _: KW_ONLY
    gamma: float = 0.99
    ir_config: IRConfig | None = None
    grad_norm_clipping: float | None = None
    batch_size: int = 64
    train_interval: tuple[int, Literal["step", "episode"]] = (5, "step")
