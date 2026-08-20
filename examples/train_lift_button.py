from marl import Experiment, algos
from marl.env import LLEConfig
from marl.nn.model_bank import qnetworks

MAP_PATH = "maps/lift3.toml"
# MAP_PATH = 2


def short_run():
    env = LLEConfig(MAP_PATH, obs_type="flattened", time_limit=32)
    trainer = algos.VDN(qnetworks.from_env(env))
    exp = Experiment.create(env, trainer, logdir="auto")
    exp.run(
        n_jobs=8,
        gpu_strategy="scatter",
        test_interval=10_000,
        save_actions=True,
        quiet=True,
        disabled_gpus=[0, 3, 4, 5, 6, 7],
    )


if __name__ == "__main__":
    short_run()
