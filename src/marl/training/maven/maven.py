from dataclasses import KW_ONLY, dataclass, field
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
from marlenv import Episode

from marl import policy
from marl.agents.hierarchical import MAVENAgent
from marl.env import EnvConfig
from marl.models import Agent, EpisodeMemory, HierarchicalTrainer, Policy, Trainer
from marl.nn.mixers import QMixMAVEN
from marl.nn.model_bank import MAVENQnetwork, qnetworks

from ..no_train import NoTrain
from ..qtarget_updater import HardUpdate, TargetParametersUpdater
from .expected_return_trainer import ExpectedReturnTrainer
from .mutual_information_trainer import MITrainer


@dataclass
class MAVEN(HierarchicalTrainer[npt.NDArray[np.int64], Trainer[npt.NDArray[np.int64]], MITrainer]):
    """
    Multi-Agent Variational ExploratioN algorithm. This algorithm is implemented as a hierarchical trainer:
        - the meta-agent is the Z-policy
        - the worker is a DQN policy that promotes the mutual information consistency between the Z-policy and the trajectories.

    Paper: https://proceedings.neurips.cc/paper_files/paper/2019/file/f816dc0acface7498e10496222e9db10-Paper.pdf
    """

    qnetwork: MAVENQnetwork
    train_policy: Policy
    env: EnvConfig
    _: KW_ONLY
    tail_type: Literal["bmm", "mul"] = "bmm"
    z_policy_type: Literal["uniform", "max-entropy", "return"] = "return"
    target_updater: TargetParametersUpdater = field(default_factory=lambda: HardUpdate(200))
    double_qlearning: bool = True
    test_policy: Policy = field(default_factory=policy.ArgMax)
    memory: EpisodeMemory = field(default_factory=lambda: EpisodeMemory(5000))
    batch_size: int = 16
    optimiser_type: Literal["adam", "rms"] = "rms"
    lr: float = 5e-4
    bandit_undiscounted: bool = True
    bandit_memory_size: int = 512
    bandit_batch_size: int = 64
    n_epochs: int = 8
    mi_loss_coef: float = 1.0
    train_interval: tuple[int, Literal["step", "episode"]] = (1, "episode")
    qmix_embed_size: int = 64
    qmix_hypernet_embed_size: int = 64

    def __post_init__(self):
        super().__post_init__()
        match self.z_policy_type:
            case "uniform":
                self.meta_trainer = NoTrain()
            case "return":
                if len(self.env.maven_bandit_obs_shape) == 1:
                    bandit_nn = qnetworks.QMLP(self.env.noise_size, self.env.maven_bandit_obs_shape, self.env.maven_bandit_extras_shape)
                elif len(self.env.maven_bandit_obs_shape) == 3:
                    bandit_nn = qnetworks.QCNN(self.env.noise_size, self.env.maven_bandit_obs_shape, self.env.maven_bandit_extras_shape)
                else:
                    raise ValueError(f"Unsupported bandit observation shape: {self.env.maven_bandit_obs_shape}")
                self.meta_trainer = ExpectedReturnTrainer(
                    bandit_nn,
                    self.env.noise_size,
                    undiscounted=self.bandit_undiscounted,
                    optimiser_type=self.optimiser_type,
                    lr=self.lr,
                    memory_size=self.bandit_memory_size,
                    batch_size=self.bandit_batch_size,
                    n_epochs=self.n_epochs,
                    train_interval=(self.train_interval[0], "episode"),
                )
            case "max-entropy":
                raise NotImplementedError("Max-entropy z policy is not implemented yet.")
        self.meta_trainer = cast(Trainer[npt.NDArray[np.int64]], self.meta_trainer)
        assert self.train_interval[1] == "episode", "MAVEN only supports training at the end of episodes."
        self.worker_trainer = MITrainer(
            self.qnetwork,
            self.train_policy,
            self.memory,
            QMixMAVEN.from_env(self.env, embed_size=self.qmix_embed_size, hypernet_embed_size=self.qmix_hypernet_embed_size),
            self.env,
            train_interval=(self.train_interval[0], "episode"),
            mi_loss_coef=self.mi_loss_coef,
            batch_size=self.batch_size,
            gamma=self.gamma,
            target_updater=self.target_updater,
            double_qlearning=self.double_qlearning,
            ir_module=self.ir_module,
            grad_norm_clipping=self.grad_norm_clipping,
            test_policy=self.test_policy,
        )
        self.name = f"MAVEN-{self.z_policy_type}_bandit"

    def update_episode(self, episode: Episode, episode_num: int, time_step: int):
        return super().update_episode(episode, episode_num, time_step)

    def make_agent(self) -> Agent[npt.NDArray[np.int64]]:
        workers = self.worker_trainer.make_agent()
        match self.z_policy_type:
            case "uniform":
                from marl.agents import RandomOneHot

                meta_agent = RandomOneHot(self.env.noise_size, n_agents=1)
            case _:
                meta_agent = self.meta_trainer.make_agent()
        return MAVENAgent(self.env.noise_size, workers, meta_agent)
