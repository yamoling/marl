---
name: marl-experiment
description: Create, run, resume, and analyse the results of MARL experiments. Use when writing or reviewing experiment scripts in this repository.
---

# MARL experiments

Use this skill when creating, running, resuming, or analysing the results of an experiment in this repository. Run commands from the repository root with `uv run python ...`.

## Core model

- **`EnvConfig`** is a serializable recipe for an environment. Call `env_config.make()` to get a fresh `MARLEnv`; its convenience properties (`n_agents`, `n_actions`, `observation_shape`, etc.) describe that environment. Use a concrete config such as `LLEConfig` in experiment scripts. For an environment under active development, implement a stable `EnvConfig` subclass rather than relying on generic pickling.
- **`Trainer`** is the serializable training-algorithm configuration. It owns trainable networks and implements updates; call `trainer.make_agent()` to construct the agent that acts in the environment. Build the trainer from the _environment config_ so its dimensions match the environment.
- An **`Experiment`** is the persistent specification: training `EnvConfig`, optional test `EnvConfig`, `Trainer`, step budget, logger choices, and experiment directory.
- A **`Run`** is one execution of that specification for one seed. Calling `experiment.run(seeds=3)` creates `run-0`, `run-1`, and `run-2`; each has independent metrics, saved actions, and (by default) checkpoints. Use multiple seeds for reportable results.

## Create and run

Use `Experiment.create`, not direct construction. It writes `experiment.json` immediately, so the experiment can be launched later from another process.

```python
from marl import Experiment, algos
from marl.env import LLEConfig
from marl.nn.model_bank.qnetworks import QMLP
from marl.nn.model_bank import qnetworks

train_env = LLEConfig(1, obs_type="layered", state_type="flattened", time_limit=78)
# Instantiate a specific Q-network
qnetwork = QMLP(
    train_env.n_actions,
    train_env.n_agents,
    train_env.observation_shape,
    train_env.extras_shape,
    duelling=False,
    independent=True,
    hidden_sizes=(16, 16),
)
# Or get a compatible Q-network from the model bank (preferred way)
qnetwork = qnetworks.from_env(train_env, recurrent=False, noisy=False, duelling=True, independent=True)
trainer = algos.DQN(qnetwork, memory_size=10, batch_size=1, train_interval=(1, "step"))

experiment = Experiment.create(
    train_env,
    trainer,
    logdir="my-dqn-baseline",
    n_steps=100_000,
    loggers=("csv",),
)
experiment.run(seeds=[0, 1, 2], test_interval=5_000, n_tests=5)
```

`test_env` defaults to a deep copy of `env`. Pass `test_env=...` explicitly when evaluation should use another map, pool, wrapper, or time limit.

For a short, executable smoke test, use [`examples/smoke_experiment.py`](examples/smoke_experiment.py). It is intentionally tiny and uses `logdir="tmp"` so it does not retain a test artifact.

### Launch an existing experiment

Creation and execution are separate. To launch or add runs to an already-created experiment, load it, then call `run`:

```python
from marl import Experiment

experiment = Experiment.load("logs/my-dqn-baseline")
experiment.run(seeds=[3, 4], test_interval=5_000, n_tests=5)
```

Do not reuse a seed whose run you want to preserve: `run-<seed>` is that seed's directory. Before adding work, inspect `experiment.runs` and `run.is_complete`.

For parallel execution, set `n_jobs`, `gpu_strategy` (`"group"` or `"scatter"`), `device`, and optionally `disabled_gpus`. Start with a single seed/job before scheduling a larger sweep.

## Storage and names

All relative log directory names are rooted under `logs/`:

| `logdir` value          | Result                           | Existing directory behaviour                                                                                                     |
| ----------------------- | -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `"baseline"`            | `logs/baseline`                  | Raises `FileExistsError` — experiment directories are unique.                                                                    |
| `Path("logs/baseline")` | `logs/baseline`                  | Raises `FileExistsError`.                                                                                                        |
| `"auto"`                | `logs/<trainer.name>-<env.name>` | Chooses this descriptive name, but does **not** make it collision-free; a pre-existing directory still raises `FileExistsError`. |
| `"tmp"` or `"test"`     | `logs/tmp` or `logs/test`        | Deletes the existing directory directly, then creates a new experiment. Use only for disposable work.                            |

Treat an experiment directory as immutable provenance. Choose a fresh explicit name for changed hyperparameters, environments, or code versions; load an existing directory only to inspect it or add intentional new-seed runs.

Expected layout with CSV logging:

```text
logs/my-dqn-baseline/
  experiment.json
  run-0/
    run.json
    train.csv
    test.csv
    training_data.csv
    test/<time_step>/          # actions and, when enabled, saved weights
```

## Logging: default to CSV

Pass `loggers=("csv",)` (the default) unless an external tracking UI is genuinely needed. CSV is local, versionable/portable, works without credentials, and has the repository's direct Polars reader.

Supported experiment logger specs are `"csv"`, `"tensorboard"`, `"wandb"`, and `"neptune"`. Multiple loggers may be requested, for example `loggers=("csv", "tensorboard")`; keep CSV included whenever later local analysis matters.

- Use **TensorBoard** for interactive scalar dashboards: `tensorboard --logdir logs`.
- Use **Weights & Biases** or **Neptune** only when the project needs their remote collaboration/dashboard features and their credentials/configuration are available. They do not provide a local metrics reader in this repository.
- Do not select `"sqlite"` through `Experiment.create`; it requires additional parameters and the run factory rejects it.

CSV writes `train.csv`, `test.csv`, and `training_data.csv` inside each run directory. Every row includes `time_step` and `timestamp_sec`. A metric schema that changes during a run causes the CSV writer to rewrite the file with the added columns, so keep metric keys stable during large runs.

## Read logs with Polars

`run.test_metrics`, `run.train_metrics`, and `run.training_data` are `polars.LazyFrame` objects, not eager dataframes. Compose `filter`, `select`, `group_by`, and other transformations first, then call `.collect()` once. This scans CSV lazily and avoids materialising unnecessary columns/rows.

```python
from pathlib import Path

import polars as pl
from marl import Experiment

experiment = Experiment.load(Path("logs/my-dqn-baseline"))
per_run = []
for run in experiment.runs:
    metrics: pl.LazyFrame = run.test_metrics
    per_run.append(
        metrics
        .filter(pl.col("time_step") >= 50_000)
        .select("time_step", "exit_rate")
        .with_columns(seed=pl.lit(run.seed))
    )

results = pl.concat(per_run).collect()
print(results)
```

For cross-seed aggregate results in repository format, use:

```python
summary = experiment.get_test_results(granularity=5_000).collect()
# Includes ticks and aggregate columns such as mean-<metric>, std-<metric>, and ci95-<metric>.
```

See [`examples/read_csv_logs.py`](examples/read_csv_logs.py) for the complete lazy-reading pattern. If no metrics were produced, Polars may raise `NoDataError` or `ColumnNotFoundError`; handle that explicitly in batch analysis.

## Useful operational checks

```python
experiment = Experiment.load("logs/my-dqn-baseline")
for run in experiment.runs:
    print(run.seed, run.is_running, run.is_complete, run.progress)

# Get a particular run either by seed or its stored run-directory string.
run_zero = experiment.get_run(0)
```

- `save_weights=True` and `save_actions=True` are the defaults for `Experiment.run`. Disable either only when storage costs outweigh checkpointing or replay needs.
- Test episodes are deterministically seeded by the runner, making per-step evaluation comparable across runs.
- To replay stored evaluation behaviour, use `experiment.replay_episode(run_seed, time_step, test_num)` when saved actions and/or checkpoints are available.
- Do not delete a named experiment merely to recreate it. The uniqueness error protects its provenance; choose a new name or deliberately use only the disposable `tmp`/`test` names.
