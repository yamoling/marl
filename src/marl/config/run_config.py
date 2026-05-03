from dataclasses import dataclass
from pathlib import Path

from marlenv import Space

from .config import Config
from .env_config import EnvConfig
from .log_config import LogConfig
from .trainer_config import TrainerConfig


@dataclass
class RunConfig[A: Space](Config):
    seed: int
    rundir: Path
    trainer: TrainerConfig
    env: EnvConfig[A]
    test_env: EnvConfig[A]
    n_steps: int
    test_interval: int
    logger: LogConfig
    save_weights: bool = True
    save_actions: bool = True
