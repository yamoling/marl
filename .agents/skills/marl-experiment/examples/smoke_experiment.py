"""Create, reload, run, and inspect a deliberately tiny disposable experiment.

Run from the repository root:
    uv run python .agents/skills/marl-experiment/examples/smoke_experiment.py

This example was smoke-tested in this repository with `n_steps=2`.
"""

import marl
from marl.env import LLEConfig
from marl.nn.model_bank import qnetworks


def main() -> None:
    env = LLEConfig(1, obs_type="flattened")
    qnetwork = qnetworks.from_env(env, recurrent=False, noisy=False, duelling=True, independent=True)
    trainer = marl.algos.DQN(qnetwork, memory_size=10, batch_size=1, train_interval=(1, "step"))

    experiment = marl.Experiment.create(env, trainer, logdir="tmp", n_steps=2, loggers=("csv",))
    print(f"Created {experiment.logdir}")

    loaded = marl.Experiment.load(experiment.logdir)
    loaded.run(test_interval=1, quiet=True)

    run = next(loaded.runs)
    print(run.test_metrics.collect())


if __name__ == "__main__":
    main()
