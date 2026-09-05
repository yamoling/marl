"""
Single-agent validation of the `ACER` trainer on discrete Gymnasium control tasks.

This is a scaled-down reproduction of Section 4 (Results on Atari) and of the ablation analysis of
Section 6.1 of the ACER paper (Wang et al., ICLR 2017, https://arxiv.org/abs/1611.01224). The Atari
suite of the paper (57 games, 16 actor-learner threads, 200M frames) is out of reach here, so the
same protocol is applied on the classic control tasks, which are single-agent and discrete just like
the domain the algorithm was designed for.

Two studies are available:

 - `--study replay` reproduces Figure 1: the replay ratio takes the values 0, 1, 4 and 8, with and
   without the trust region update. A replay ratio of 0 is the purely on-policy actor-critic
   baseline (the paper's A3C control), so the expected outcome is a monotone (with diminishing
   returns) improvement of the sample efficiency as the replay ratio grows.
 - `--study ablation` reproduces Figure 4: ACER is compared against variants where a single
   component is removed, i.e. the Retrace correction (`retrace_threshold`, 0 = one-step target,
   +inf = uncorrected Q(lambda)/importance sampling), the truncation with bias correction
   (`truncation_threshold=+inf`) and the trust region.

Everything runs on CPU: the experiments are small and parallelised over the configurations and the
seeds rather than over the environment steps.

Usage:
    python scripts/train_acer_gym.py --study replay --env-id CartPole-v1
    python scripts/train_acer_gym.py --study ablation --env-id Acrobot-v1
"""

import multiprocessing as mp
import os
import sys
from dataclasses import dataclass
from typing import Literal

import typed_argparse as tap

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

STUDIES = ("replay", "ablation")
Study = Literal["replay", "ablation"]

INFINITE = 1e9
"""Stand-in for an infinite truncation threshold (i.e. no truncation at all)."""


@dataclass
class Config:
    """One point of the study: a human-readable name and the ACER keyword arguments it overrides."""

    name: str
    kwargs: dict


def replay_configs() -> list[Config]:
    """
    Configurations of Figure 1 of the paper: replay ratios 0, 1, 4 and 8, with and without the trust
    region update. A replay ratio of 0 disables experience replay entirely and is the on-policy
    actor-critic baseline.

    @ai-generated
    """
    configs = []
    for ratio in (0.0, 1.0, 4.0, 8.0):
        for trust_region in (True, False):
            suffix = "trust" if trust_region else "no-trust"
            configs.append(Config(f"replay-{ratio:g}-{suffix}", {"replay_ratio": ratio, "trust_region": trust_region}))
    return configs


def ablation_configs() -> list[Config]:
    """
    Configurations of Figure 4 of the paper: full ACER against variants missing one component each.

    @ai-generated
    """
    return [
        Config("acer", {}),
        Config("no-trust-region", {"trust_region": False}),
        Config("no-truncation", {"truncation_threshold": INFINITE}),
        Config("one-step-target", {"retrace_threshold": 0.0}),
        Config("uncorrected-q-lambda", {"retrace_threshold": INFINITE}),
    ]


def make_runs(args: "Args", config: Config):
    """
    Build the `Run` objects of one configuration: an `Experiment` on the requested Gymnasium task
    with an MLP actor and an MLP critic, and one run per seed.

    @ai-generated
    """
    from marlenv.adapters.gym_adapter import Gym

    from marl import Experiment, algos
    from marl.env import EnvConfig
    from marl.nn.model_bank import actor_critics, qnetworks
    from marl.utils import Schedule

    env = EnvConfig.from_any(Gym(args.env_id), agent_id=False)
    actor, _ = actor_critics.from_env(env, recurrent=False, actor_kwargs={"mlp_sizes": (64, 64)})
    critic = qnetworks.from_env(env, duelling=False, hidden_sizes=(64, 64))
    # Hyper-parameters of Section 4 of the paper, overridden by the configuration of the study.
    kwargs = {
        "gamma": 0.99,
        "lr_actor": args.lr_actor,
        "lr_critic": args.lr_critic,
        "entropy_coef": Schedule.constant(1e-3),
        "truncation_threshold": 10.0,
        "trust_region_delta": 1.0,
        "trust_region_decay": 0.99,
        "batch_size": args.batch_size,
        "memory_size": args.memory_size,
        "replay_start": args.replay_start,
        "grad_norm_clipping": 10.0,
    } | config.kwargs
    trainer = algos.ACER(actor, critic, None, **kwargs)
    logdir = f"acer-{args.study}/{args.env_id}/{config.name}"
    experiment = Experiment.create(env, trainer, logdir=logdir, n_steps=args.n_steps)
    return experiment.create_runs(
        seeds=range(args.n_seeds),
        n_tests=args.n_tests,
        test_interval=args.test_interval,
        save_weights=False,
        save_actions=False,
    )


def run_all(runs: list, n_jobs: int, device: str = "cpu"):
    """
    Run every `Run` on `device`, `n_jobs` at a time.

    `marl.runners.parallel_run` is not used because it always places the first run on the automatic
    (i.e. GPU) device, which the small networks of these studies do not need. The extra
    `round-robin` device deals the runs over the visible GPUs in turn, which is deterministic,
    unlike the `scatter` strategy whose free-memory readings race when the workers all start at once.

    @ai-generated
    """
    import torch

    from marl.runners.parallel_runner import _start_run

    devices = [device] * len(runs)
    if device == "round-robin":
        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            raise RuntimeError("No GPU available for the round-robin device assignment.")
        devices = [index % n_gpus for index in range(len(runs))]
    kwargs = [
        {
            "run": run,
            "device_type": run_device,
            "quiet": True,
            "render_tests": False,
            "estimated_gpu_memory": 0,
            "auto_device_strategy": "group",
        }
        for run, run_device in zip(runs, devices)
    ]
    with mp.get_context("spawn").Pool(n_jobs, maxtasksperchild=1) as pool:
        handles = [pool.apply_async(_start_run, kwds=kwds) for kwds in kwargs]
        for handle, kwds in zip(handles, kwargs):
            try:
                handle.get()
            except Exception as error:  # noqa: BLE001
                print(f"Run {kwds['run'].rundir} failed: {error}", file=sys.stderr)


class Args(tap.TypedArgs):
    study: Study = tap.arg("--study", default="replay", help="Which study to run.")
    env_id: str = tap.arg("--env-id", default="CartPole-v1")
    n_steps: int = tap.arg("--n-steps", default=200_000)
    n_seeds: int = tap.arg("--n-seeds", default=5)
    n_jobs: int = tap.arg("--n-jobs", default=20)
    test_interval: int = tap.arg("--test-interval", default=5_000)
    n_tests: int = tap.arg("--n-tests", default=10)
    lr_actor: float = tap.arg("--lr-actor", default=5e-4)
    lr_critic: float = tap.arg("--lr-critic", default=1e-3)
    batch_size: int = tap.arg("--batch-size", default=8)
    memory_size: int = tap.arg("--memory-size", default=2_000)
    replay_start: int = tap.arg("--replay-start", default=32)
    dry_run: bool = tap.arg("--dry-run", default=False)


def main(args: Args):
    """
    Create every configuration of the requested study and train them all.

    @ai-generated
    """
    configs = replay_configs() if args.study == "replay" else ablation_configs()
    runs = []
    for config in configs:
        runs += make_runs(args, config)
    print(f"{len(configs)} configurations x {args.n_seeds} seeds = {len(runs)} runs of {args.n_steps} steps")
    for config in configs:
        print(f" - {config.name}: {config.kwargs}")
    if args.dry_run:
        return
    run_all(runs, args.n_jobs)


if __name__ == "__main__":
    tap.Parser(Args).bind(main).run()
