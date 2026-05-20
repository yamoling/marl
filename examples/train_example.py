from marl import Experiment, algos
from marl.env import LLEConfig
from marl.models import TransitionMemory
from marl.nn import mixers
from marl.nn.model_bank import qnetworks
from marl.policy import EpsilonGreedy


def mulitple_parallel_runs():
    env = LLEConfig(6, obs_type="layered")
    trainer = algos.VDN(
        qnetworks.from_env(env, independent=True),
        memory_size=50_000,
        train_policy=EpsilonGreedy.linear(1, 0.05, 100_000),
        gamma=0.95,
        train_interval=(5, "step"),
        lr=5e-4,
        batch_size=64,
        optimiser_type="adam",
        grad_norm_clipping=10,
    )
    exp = Experiment(env, trainer, logdir="auto")
    exp.run(seeds=8, test_interval=5000, gpu_strategy="scatter", n_jobs=8, disabled_gpus=range(6), quiet=True)


def short_run():
    env = LLEConfig(6, obs_type="flattened")
    trainer = algos.QMix(qnetworks.from_env(env), memory_size=50_000, mixer=mixers.QMix.from_env(env))
    exp = Experiment(env, trainer, logdir="auto", n_steps=5_000)
    exp.run(test_interval=500)


if __name__ == "__main__":
    short_run()
    mulitple_parallel_runs()
