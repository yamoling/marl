# Experiment Runner Guide

Purpose
- Describe how to run experiments reproducibly with the repository's runner scripts, recommended flags, GPU selection, outputs, and best practices.

Quickstart
- Basic run (example):

```bash
# create a logs/ entry and run an experiment
python src/start_run.py --config configs/my_experiment.toml --output logs/my_experiment

# or generate many experiments programmatically
python src/create_experiments.py --template configs/experiment_template.toml --outdir logs/batch_experiments
```

Note: `start_run.py` and `create_experiments.py` are the canonical entry points. Their exact flags may vary; prefer using TOML experiment configs as inputs and record any CLI overrides.

Runner responsibilities
- Parse experiment config (TOML), create experiment directory, initialize trainer and environment, run training loop, checkpoint, and write `experiment.json` metadata.
- Keep side effects local to the experiment output folder.

Recommended CLI and env vars
- Prefer config-first runs: store full experiment configuration in a TOML file and pass only the path on the CLI.
- Record overrides: when passing CLI flags, echo the full command into the experiment folder (e.g., `cmd.txt`).
- GPU selection: use `CUDA_VISIBLE_DEVICES` to assign GPUs. Runners should support an explicit `--gpu` or honor `CUDA_VISIBLE_DEVICES`.

GPU selection patterns
- Manual (simple):

```bash
CUDA_VISIBLE_DEVICES=0 python src/start_run.py --config configs/my_experiment.toml
```

- Automatic (pick first free GPU):

```bash
# pick GPU with lowest memory usage (requires nvidia-smi)
GPU=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print $1 " " NR-1}' | sort -n | head -n1 | awk '{print $2}')
CUDA_VISIBLE_DEVICES=$GPU python src/start_run.py --config configs/my_experiment.toml
```

- Multi-GPU: limit to 4 GPUs per distributed job. Prefer coordinated orchestration (slurm, k8s) rather than ad-hoc local scattering.

Outputs and artifact layout
- `logs/<experiment_name>/run_<timestamp>_seed=<N>/`
  - `checkpoints/`: model checkpoints (use clear versioned names, e.g., `ckpt_step=10000.pt`).
  - `logs/`: stdout/stderr and structured logs (JSON lines, CSV summary, TensorBoard dir as applicable).
  - `experiment.json`: canonical metadata (see Metadata section below).
  - `cmd.txt`: command line used to run the job.

Checkpointing and resuming
- Save checkpoints atomically (write to a temporary file and rename) to avoid corrupted files on interruption.
- Include optimizer state and RNG states in checkpoints for exact reproducibility.
- Provide a `--resume` flag in the runner to resume from a checkpoint path.

Logging and observability
- Structured logs: write metrics in CSV and/or JSON lines for parsing. Also write TensorBoard scalars if used.
- Keep raw console output saved as `stdout.txt`/`stderr.txt` in the run folder.

Metadata (`experiment.json`) — recommended fields
- `name`: short experiment name
- `timestamp`: ISO-8601 start time
- `seed`: integer RNG seed
- `config_path`: path or copy of the TOML used
- `cli`: full CLI command string
- `git_commit`: commit hash (or diff/patch) capturing code state
- `artifacts`: list of saved artifact URIs (checkpoints, logs)

Reproducibility checklist
- Save the full TOML config used for the run in the run folder.
- Record seed(s) and RNG state when saving checkpoints.
- Record `git_commit` or a patch file in the run folder.
- Record the exact command in `cmd.txt` and the environment variables used.

Best practices and tips
- Avoid writing large blobs into the repo. Use external artifact storage and record URIs in `experiment.json`.
- Keep experiments small and composable: use `create_experiments.py` to generate repeatable batches.
- Test runner changes with a short smoke test (small steps, tiny environments) before full-scale runs.
- Use consistent naming conventions for experiments and checkpoints.

Troubleshooting
- Out of memory: reduce batch size or use gradient accumulation; log GPU memory on failure.
- Corrupted checkpoints: ensure atomic writes and verify checksum after save.

Extending the runner
- Add a clear `Runner` or `Experiment` class that encapsulates lifecycle: setup, train loop, checkpoint, teardown.
- Keep CLI parsing shallow — delegate experiment creation to factory functions that accept a parsed config object.

Examples and common commands
- Start a single run with a chosen seed:

```bash
CUDA_VISIBLE_DEVICES=0 python src/start_run.py --config configs/my_experiment.toml --seed 42 --output logs/my_experiment
```

- Generate and queue 10 experiments programmatically:

```bash
python src/create_experiments.py --template configs/experiment_template.toml --count 10 --outdir logs/batch_experiments
```

Where to put this guide
- Keep this file in `doc/` and link it from `README.md`.
