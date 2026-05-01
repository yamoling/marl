from dataclasses import KW_ONLY, dataclass, field
from typing import Literal, cast
import torch

import numpy as np
import numpy.typing as npt
from marlenv import Episode

from marl.agents.hierarchical import MAVENAgent
from marl.models import Agent, IRModule, Mixer, Policy, QNetwork, Batch, EpisodeMemory, HierarchicalTrainer, Trainer

from .expected_return_trainer import ExpectedReturnTrainer
from .mutual_information_trainer import MITrainer
from ..no_train import NoTrain
from ..qtarget_updater import TargetParametersUpdater, SoftUpdate


@dataclass
class MAVEN(HierarchicalTrainer[npt.NDArray[np.int64], Trainer[npt.NDArray[np.int64]], MITrainer]):
    """
    Multi-Agent Variational ExploratioN algorithm. This algorithm is implemented as a hierarchical trainer:
        - the meta-agent is the Z-policy
        - the worker is a DQN policy that promotes the mutual information consistency between the Z-policy and the trajectories.

    Paper: https://proceedings.neurips.cc/paper_files/paper/2019/file/f816dc0acface7498e10496222e9db10-Paper.pdf
    """

    qnetwork: QNetwork
    train_policy: Policy
    noise_size: int
    n_actions: int
    n_agents: int
    state_size: int
    state_extras_size: int
    mixer: Mixer
    _: KW_ONLY
    z_policy_type: Literal["uniform", "max-entropy", "return"] = "return"
    return_bandit_nn: QNetwork | None = None
    batch_size: int = 64
    gamma: float = 0.99
    target_updater: TargetParametersUpdater = field(default_factory=lambda: SoftUpdate(1e-2))
    double_qlearning: bool = True
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
                self.meta_trainer = NoTrain()
            case "return":
                assert self.return_bandit_nn is not None, "return_bandit_nn must be provided when z_policy_type is 'return'"
                self.meta_trainer = ExpectedReturnTrainer(
                    self.return_bandit_nn,
                    self.noise_size,
                    undiscounted=self.undiscounted,
                    optimiser_type=self.optimiser_type,
                    lr=self.lr,
                    memory_size=self.bandit_memory_size,
                    batch_size=self.bandit_batch_size,
                    n_epochs=self.n_epochs,
                )
            case "max-entropy":
                raise NotImplementedError("Max-entropy z policy is not implemented yet.")
        self.meta_trainer = cast(Trainer[npt.NDArray[np.int64]], self.meta_trainer)
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
        self.name = f"{self.__class__.__name__}-{self.mixer.name}-{self.z_policy_type}"

    def update_episode(self, episode: Episode, episode_num: int, time_step: int):
        return super().update_episode(episode, episode_num, time_step)

    def get_mixing_kwargs(self, batch: Batch, all_qvalues: torch.Tensor, is_next: bool):
        return {"maven_noise": batch["maven-noise"]}

    def make_agent(self) -> Agent[npt.NDArray[np.int64]]:
        workers = self.worker_trainer.make_agent()
        match self.z_policy_type:
            case "uniform":
                from marl.agents import RandomOneHot

                meta_agent = RandomOneHot(self.noise_size, n_agents=1)
            case _:
                meta_agent = self.meta_trainer.make_agent()
        return MAVENAgent(self.noise_size, workers, meta_agent)
