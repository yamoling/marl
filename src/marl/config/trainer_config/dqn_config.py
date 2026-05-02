from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from marlenv import MARLEnv

from marl.config.nn_config import NetworkConfig

from ..memory_config import MemoryConfig
from ..policy_config import PolicyConfig
from ..target_updater_config import TargetUpdaterConfig
from .trainer_config import TrainerConfig

if TYPE_CHECKING:
    from marl import Trainer


@dataclass
class DQNConfig(TrainerConfig):
    qnetwork: NetworkConfig
    train_policy: PolicyConfig
    memory: MemoryConfig
    optimiser_type: Literal["adam", "rmsprop"] = "adam"
    lr: float = 1e-4
    target_updater: TargetUpdaterConfig = field(default_factory=TargetUpdaterConfig.default)
    double_qlearning: bool = True
    test_policy: PolicyConfig | None = None

    def make(self, env: MARLEnv | None = None) -> Trainer:
        from marl.training import DQN

        return DQN(
            qnetwork=self.qnetwork.make_qnetwork(),
            train_policy=self.train_policy.make(),
            memory=self.memory.make(),
            optimiser_type=self.optimiser_type,
            gamma=self.gamma,
            batch_size=self.batch_size,
            lr=self.lr,
            train_interval=self.train_interval,
            target_updater=self.target_updater.make(),
            double_qlearning=self.double_qlearning,
            grad_norm_clipping=self.grad_norm_clipping,
            test_policy=self.test_policy.make() if self.test_policy is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
