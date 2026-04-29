from dataclasses import KW_ONLY, dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
from marlenv import Episode
from marlenv.catalog import DiscreteMockEnv

from marl.agents.hierarchical import MAVENAgent
from marl.models import IRModule, Mixer, Policy, QNetwork
from marl.models.replay_memory.biased_memory import EpisodeMemory
from marl.models.trainer import HierarchicalTrainer, Trainer
from marl.training.no_train import NoTrain
from marl.training.qtarget_updater import SoftUpdate, TargetParametersUpdater

from .expected_return_trainer import ExpectedReturnTrainer
from .mutual_information_trainer import MITrainer


@dataclass
class MAVEN(HierarchicalTrainer[npt.NDArray[np.int64], Trainer, MITrainer]):
    """
    Multi-Agent Variational ExploratioN algorithm. This algorithm is implemented as a hierarchical trainer:
        - the meta-agent is the Z-policy
        - the worker is a DQN policy that promotes the mutual information consistency between the Z-policy and the trajectories.

    Paper: https://proceedings.neurips.cc/paper_files/paper/2019/file/f816dc0acface7498e10496222e9db10-Paper.pdf
    """

    qnetwork: QNetwork
    train_policy: Policy
    z_policy_type: Literal["uniform", "max-entropy", "return"]
    return_bandit_nn: QNetwork
    noise_size: int
    n_actions: int
    n_agents: int
    state_size: int
    state_extras_size: int
    _: KW_ONLY
    batch_size: int = 64
    gamma: float = 0.99
    target_updater: TargetParametersUpdater = field(default_factory=lambda: SoftUpdate(1e-2))
    double_qlearning: bool = True
    mixer: Mixer | None = None
    ir_module: IRModule | None = None
    grad_norm_clipping: float | None = None
    test_policy: Policy | None = None
    memory_size: int = 5_000
    undiscounted: bool = True
    optimiser_type: Literal["adam", "rms"] = "adam"
    lr: float = 1e-5
    bandit_memory_size: int = 512
    bandit_batch_size: int = 64
    n_epochs: int = 8
    train_interval: tuple[int, Literal["episode"]] = (1, "episode")
    mi_loss_coef: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        match self.z_policy_type:
            case "uniform":
                # The DiscreteMockEnv is there because we need a placeholder. The agent will never be instanciated from here, though.
                self.meta_trainer = NoTrain.discrete(DiscreteMockEnv())
            case "return":
                self.meta_trainer = ExpectedReturnTrainer(
                    self.return_bandit_nn,
                    undiscounted=self.undiscounted,
                    optimiser_type=self.optimiser_type,
                    lr=self.lr,
                    memory_size=self.bandit_memory_size,
                    batch_size=self.bandit_batch_size,
                    n_epochs=self.n_epochs,
                )
            case "max-entropy":
                raise NotImplementedError("Max-entropy z policy is not implemented yet.")
        self.worker_trainer = MITrainer(
            self.qnetwork,
            self.train_policy,
            EpisodeMemory(self.memory_size),
            self.noise_size,
            self.n_actions,
            self.n_agents,
            self.state_size,
            self.state_extras_size,
            train_interval=self.train_interval,
            mi_loss_coef=self.mi_loss_coef,
            batch_size=self.batch_size,
            gamma=self.gamma,
            target_updater=self.target_updater,
            double_qlearning=self.double_qlearning,
            mixer=self.mixer,
            ir_module=self.ir_module,
            grad_norm_clipping=self.grad_norm_clipping,
            test_policy=self.test_policy,
        )

    def update_episode(self, episode: Episode, episode_num: int, time_step: int):
        return super().update_episode(episode, episode_num, time_step)

    def make_agent(self):
        workers = self.worker_trainer.make_agent()
        bandit = self.meta_trainer.make_agent()
        return MAVENAgent(self.noise_size, workers, bandit)
