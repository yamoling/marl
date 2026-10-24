"""Seeds 8-15 of the BNAIC 2023 LLE baselines (IQL/QMIX, levels 1-5) for the LLE overview figure.

Run from the repository root: .venv/bin/python scripts/bnaic2023-remaining-seeds.py
Only the `experiment.json` files are needed. Existing run directories are never overwritten.

@ai-generated
"""

import logging
from pathlib import Path

from marl import Experiment
from marl.runners import parallel_run

logger = logging.getLogger(__name__)

ROOT = Path("logs/bnaic2023-baselines-eps200k")
DIRECTORIES = [
    "lle1-iql-500000",
    "lle1-qmix-500000",
    "lle2-iql-500000",
    "lle2-qmix-500000",
    "lle3-iql-750000",
    "lle3-qmix-750000",
    "lle4-iql-750000",
    "lle4-qmix-750000",
    "lle5-iql-1000000",
    "lle5-qmix-1000000",
]
SEEDS = range(8, 16)
N_JOBS = 8
# Identical in the run.json of seeds 0-7 of every experiment (checked on 2026-10-24).
RUN_SETTINGS = {"n_tests": 5, "test_interval": 5_000, "save_weights": True, "save_actions": True, "save_training_data": False}


def main():
    """Queue the missing seeds of every experiment in one shared worker pool. @ai-generated"""
    runs = []
    for name in DIRECTORIES:
        exp = Experiment.load(ROOT / name)
        existing = {run.seed for run in exp.runs if run.is_running or run.is_complete}
        seeds = [s for s in SEEDS if s not in existing]
        for seed in seeds:
            if (exp.logpath / f"run-{seed}").exists():
                raise FileExistsError(f"{exp.logpath}/run-{seed} already exists but is neither complete nor running.")
        logger.info("%s: queueing seeds %s", name, seeds)
        runs += exp.create_runs(seeds, **RUN_SETTINGS)
    logger.info("Launching %s runs with %s parallel jobs.", len(runs), N_JOBS)
    parallel_run(runs, n_jobs=N_JOBS, device="auto", gpu_strategy="scatter", quiet=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
