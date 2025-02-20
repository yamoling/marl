from typing import Literal
from lle import LLE
import marlenv
from marl.training import DQNTrainer, SoftUpdate
from marl.training.intrinsic_reward import AdvantageIntrinsicReward
from marl.training.continuous_ppo_trainer import ContinuousPPOTrainer
from marl.training.haven_trainer import HavenTrainer
from marl.policy import EpsilonGreedy, ArgMax
from marl.training.mixers import VDN
from marl.nn.model_bank.actor_critics import CNNContinuousActorCritic
from marl.utils import MultiSchedule, Schedule
from marl.models import TransitionMemory

import marl

env = LLE.from_file("maps/2b-quad.toml").obs_type("layered").build()
env = marlenv.Builder(env).time_limit(28).agent_id().build()

trainer = DQNTrainer(
    marl.nn.model_bank.qnetworks.CNN.from_env(env),
    EpsilonGreedy.linear(1.0, 0.05, 100_000),
    TransitionMemory(50_000),
    mixer=VDN(env.n_agents),
    double_qlearning=True,
    gamma=0.95,
    test_policy=ArgMax(),
)

trainer.make_agent()
exp = marl.Experiment.create(env, 300_000, trainer=trainer, test_interval=5000, logdir="2b-quad")
# exp.run(n_tests=10)
