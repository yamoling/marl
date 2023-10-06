import rlenv as rl
from lle import LLE, ObservationType
from marl.models import Experiment
from marl.qlearning.dqn_nodes import DQN
from marl.training import DQNTrainer
from marl.qlearning import mixers
from marl.nn.model_bank import CNN

import marl


def load(experiment_dir: str):
    exp = Experiment.load(experiment_dir)
    runner = exp.create_runner("csv")
    runner.train(5)


def create(experiment_dir: str):
    env = rl.Builder(LLE.level(6, ObservationType.LAYERED)).agent_id().time_limit(78).build()
    ir = marl.intrinsic_reward.RandomNetworkDistillation(env.observation_shape, env.extra_feature_shape)
    trainer = DQNTrainer(
        qnetwork=CNN.from_env(env),
        lr=1e-3,
        mixer=mixers.VDN(env.n_agents),
        ir_module=ir,
        double_qlearning=True,
        update_frequency="step",
        train_interval=5,
    )
    trainer.show()
    dqn = DQN(trainer)

    exp = Experiment.create(experiment_dir, dqn, env, 10_000, 500)
    runner = exp.create_runner("csv")
    runner.train(5)


if __name__ == "__main__":
    directory = "logs/test"
    create(directory)
    # load(directory)
