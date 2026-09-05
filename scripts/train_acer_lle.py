"""
Multi-agent validation of the `ACER` trainer on the Laser Learning Environment.

The ACER paper is single-agent, so this script checks the multi-agent extension implemented in
`marl.algos.ACER` against the algorithms that are already known to work in this repository:

 - `iacer` / `macer-vdn` / `macer-qmix`: ACER without a mixer (independent learners) and with the two
   usual mixers, i.e. the centralised variant where Retrace runs on the joint action-value function;
 - `iacer-onpolicy`: the same as `iacer` with a replay ratio of 0, which removes experience replay
   altogether. Comparing it with `iacer` tells whether the off-policy corrections actually buy
   sample efficiency in the multi-agent case, which is the central claim of the paper;
 - `ippo` / `mappo-qmix`: the PPO baselines of the repository, trained on the same environment for
   the same number of steps, as a reference point.

Note that the joint importance weight of the centralised variant is the product of the per-agent
weights, so its variance grows with the number of agents: the `acer/mean-importance-weight` column of
`training_data.csv` is the diagnostic to look at if the centralised runs behave worse than the
decentralised ones.

Usage:
    python scripts/train_acer_lle.py --level 3 --n-steps 500000 --device cpu
"""

import os
import sys
from typing import Literal

import typed_argparse as tap

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

Condition = Literal["iacer", "iacer-onpolicy", "macer-vdn", "macer-qmix", "ippo", "mappo-qmix"]
CONDITIONS: tuple[Condition, ...] = ("iacer", "iacer-onpolicy", "macer-vdn", "macer-qmix", "ippo", "mappo-qmix")


def make_trainer(condition: Condition, env, args: "Args"):
    """
    Build the trainer of one condition. Every condition shares the same actor and critic architecture
    so that the comparison only reflects the learning algorithm.

    @ai-generated
    """
    from marl import algos
    from marl.nn import mixers
    from marl.nn.model_bank import actor_critics, qnetworks
    from marl.utils import Schedule

    recurrent = False
    actor, critic = actor_critics.from_env(env, recurrent, independent=False)
    match condition:
        case "iacer" | "iacer-onpolicy":
            mixer = None
        case "macer-vdn":
            mixer = mixers.VDN.from_env(env)
        case "macer-qmix":
            mixer = mixers.QMix.from_env(env)
        case "ippo":
            mixer = None
        case "mappo-qmix":
            mixer = mixers.QMix.from_env(env)
    if condition in ("ippo", "mappo-qmix"):
        return algos.PPO(
            actor,
            critic,
            mixer=mixer,
            gamma=args.gamma,
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            train_interval=(128, "step"),
            minibatch_size=32,
            n_epochs=10,
            c2=Schedule.constant(0.01),
            grad_norm_clipping=10.0,
        )
    # ACER needs a Q-network critic (one utility per action) instead of the PPO state-value critic.
    acer_critic = qnetworks.from_env(env, recurrent=recurrent, independent=False, duelling=False)
    return algos.ACER(
        actor,
        acer_critic,
        mixer,
        gamma=args.gamma,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        replay_ratio=0.0 if condition == "iacer-onpolicy" else args.replay_ratio,
        batch_size=args.batch_size,
        memory_size=args.memory_size,
        entropy_coef=Schedule.constant(0.01),
        truncation_threshold=10.0,
        trust_region=True,
        trust_region_delta=1.0,
        trust_region_decay=0.99,
        grad_norm_clipping=10.0,
    )


def make_runs(args: "Args", condition: Condition):
    """
    Create the experiment of one condition and its per-seed runs.

    @ai-generated
    """
    from marl import Experiment
    from marl.env import LLEConfig

    env = LLEConfig(args.level, obs_type=args.obs_type, state_type="flattened")
    trainer = make_trainer(condition, env, args)
    logdir = f"acer-lle{args.level}/{condition}"
    experiment = Experiment.create(env, trainer, logdir=logdir, n_steps=args.n_steps)
    return experiment.create_runs(
        seeds=range(args.n_seeds),
        n_tests=args.n_tests,
        test_interval=args.test_interval,
        save_weights=False,
        save_actions=False,
    )


class Args(tap.TypedArgs):
    level: int = tap.arg("--level", default=3)
    obs_type: str = tap.arg("--obs-type", default="layered", help="'layered' (CNN) or 'flattened' (MLP).")
    conditions: list[str] = tap.arg("--conditions", default=list(CONDITIONS), nargs="+")
    n_steps: int = tap.arg("--n-steps", default=500_000)
    n_seeds: int = tap.arg("--n-seeds", default=4)
    n_jobs: int = tap.arg("--n-jobs", default=8)
    device: str = tap.arg("--device", default="cpu", help="'cpu', 'auto' or a cuda device index.")
    test_interval: int = tap.arg("--test-interval", default=10_000)
    n_tests: int = tap.arg("--n-tests", default=10)
    gamma: float = tap.arg("--gamma", default=0.95)
    lr_actor: float = tap.arg("--lr-actor", default=5e-4)
    lr_critic: float = tap.arg("--lr-critic", default=1e-3)
    replay_ratio: float = tap.arg("--replay-ratio", default=4.0)
    batch_size: int = tap.arg("--batch-size", default=8)
    memory_size: int = tap.arg("--memory-size", default=2_000)
    dry_run: bool = tap.arg("--dry-run", default=False)


def main(args: Args):
    """
    Train every requested condition on the requested LLE level.

    @ai-generated
    """
    from train_acer_gym import run_all

    runs = []
    for condition in args.conditions:
        runs += make_runs(args, condition)  # pyright: ignore[reportArgumentType]
    print(f"{len(args.conditions)} conditions x {args.n_seeds} seeds = {len(runs)} runs of {args.n_steps} steps")
    if args.dry_run:
        return
    run_all(runs, args.n_jobs, device=args.device)


if __name__ == "__main__":
    tap.Parser(Args).bind(main).run()
