from marlenv import catalog

from marl import Experiment
from marl.env import EnvConfig, LLEConfig
from marl.models import EpisodeMemory
from marl.nn.model_bank import MAVENQnetwork, qnetworks
from marl.policy import EpsilonGreedy
from marl.training import MAVEN, QMix

if __name__ == "__main__":
    env = EnvConfig.from_any(catalog.MStepsMatrix(10), maven_noise_size=16)
    # env = LLEConfig(6, maven_noise_size=16)
    experiment = Experiment(
        env,
        MAVEN(
            MAVENQnetwork.from_env(env),
            EpsilonGreedy.linear(1, 0.01, 100),
            env,
            train_interval=(1, "episode"),
            grad_norm_clipping=10.0,
            batch_size=32,
        ),
        150_000,
        "test",
    )
    # trainer = experiment.trainer
    # print(trainer.qnetwork)
    # print(trainer.meta_trainer.nn)
    # print(trainer.worker_trainer.qnetwork)
    # print(trainer.worker_trainer.mixer)
    experiment.run(seeds=12, test_interval=2500, n_jobs=12, disabled_gpus=[0, 1, 2, 3], gpu_strategy="scatter")
