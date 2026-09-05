"""Reproduce LAN's 14-map SMAC benchmark and optional LAN-mean ablation.

Run on the experiment machine: uv run python scripts/reproduce_lan.py --run
Without --run this only creates experiment specifications.
"""

import argparse
from pathlib import Path

from marl import Experiment
from marl.algos import LAN
from marl.env.config.smac_config import SMACConfig

MAPS = (
    "2s3z",
    "3s5z",
    "1c3s5z",
    "5m_vs_6m",
    "10m_vs_11m",
    "27m_vs_30m",
    "3s5z_vs_3s6z",
    "MMM2",
    "2s_vs_1sc",
    "3s_vs_5z",
    "6h_vs_8z",
    "bane_vs_bane",
    "2c_vs_64zg",
    "corridor",
)


def main():
    """Create immutable per-map specifications and optionally launch seeded runs. @ai-generated"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", nargs="+", choices=MAPS, default=list(MAPS))
    parser.add_argument("--variants", nargs="+", choices=("lan", "lan-mean"), default=["lan"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--log-prefix", default="lan-paper-v3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--game-version", default="4.6.2")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--load", action="store_true", help="Load existing specifications and add only unused seeds")
    args = parser.parse_args()
    if args.jobs < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("jobs must be positive and seeds must be unique")
    if args.log_prefix in ("tmp", "test") or Path(args.log_prefix).is_absolute() or ".." in Path(args.log_prefix).parts:
        parser.error("Use a persistent relative log prefix without '..'")
    for variant in args.variants:
        for map_name in args.maps:
            directory = Path("logs") / args.log_prefix / variant / map_name
            if args.load:
                experiment = Experiment.load(directory)
                if not isinstance(experiment.trainer, LAN) or not isinstance(experiment.env, SMACConfig):
                    raise ValueError(f"Not a LAN/SMAC experiment: {directory}")
                if (
                    experiment.env.game_version != args.game_version
                    or experiment.env.map_name != map_name
                    or experiment.trainer.qnetwork.mean_center != (variant == "lan-mean")
                ):
                    raise ValueError(f"Stored configuration does not match requested experiment: {directory}")
            else:
                env = SMACConfig(map_name, game_version=args.game_version, agent_id=True, last_action=True)
                trainer = LAN.from_env(env, mean_center=variant == "lan-mean")
                experiment = Experiment.create(env, trainer, logdir=directory, n_steps=2_000_000, loggers=("csv",))
            print(f"{variant}: {map_name} -> {directory}")
            if args.run:
                existing = {run.seed for run in experiment.runs}
                seeds = [seed for seed in args.seeds if seed not in existing]
                if not seeds:
                    print("All requested seeds already exist; leaving their results intact.")
                    continue
                experiment.run(
                    seeds=seeds,
                    device=args.device,
                    n_jobs=args.jobs,
                    n_tests=32,
                    test_interval=10_000,
                    save_weights=True,
                    save_actions=True,
                )


if __name__ == "__main__":
    main()
