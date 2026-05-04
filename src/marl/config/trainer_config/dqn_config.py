from dataclasses import KW_ONLY, dataclass, field
from typing import Literal, override

from marl.config.nn_config import QNetworkConfig
from marl.training import DQN

from ..memory_config import MemoryConfig
from ..mixer_config import MixerConfig
from ..policy_config import PolicyConfig
from ..target_updater_config import TargetUpdaterConfig
from .trainer_config import TrainerConfig


@dataclass
class DQNConfig(TrainerConfig[DQN]):
    qnetwork: QNetworkConfig
    train_policy: PolicyConfig
    memory: MemoryConfig
    _: KW_ONLY
    mixer: MixerConfig | None = None
    optimiser_type: Literal["adam", "rmsprop"] = "adam"
    lr: float = 1e-4
    target_updater: TargetUpdaterConfig = field(default_factory=TargetUpdaterConfig.default)
    double_qlearning: bool = True
    test_policy: PolicyConfig | None = None

    @override
    def make(self):
        from marl.training import DQN

        mixer = None
        if self.mixer is not None:
            mixer = self.mixer.make()
        return DQN(
            qnetwork=self.qnetwork.make(),
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
            mixer=mixer,
        )
