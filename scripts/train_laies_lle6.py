"""Train LAIES on LLE level 6 with the hyperparameters of the LAIES paper.

LAIES (Liu et al., ICML 2023, https://proceedings.mlr.press/v202/liu23ac.html) is QMIX
augmented with individual (IDI) and collaborative (CDI) diligence intrinsic rewards
computed from an external-state transition model. The QMIX side therefore uses the
PyMARL configuration reported in the paper: recurrent agent networks with shared
parameters, RMSProp at 5e-4, batches of 32 episodes, a 5,000-episode replay buffer,
a hard target update every 200 episodes, gamma = 0.99 and gradient clipping at 10.
Observations are LLE's layered maps encoded by a CNN before the recurrent head, as is
customary in this repository, and the global state is flattened as LAIES requires.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Literal

import dotenv
import typed_argparse as tap

import marl
from marl.algos import LAIES, HardUpdate
from marl.env import LLEConfig
from marl.nn import mixers
from marl.nn.model_bank import qnetworks
from marl.policy import ArgMax, EpsilonGreedy

LEVEL = 6
LOGGER = logging.getLogger(__name__)


class Args(tap.TypedArgs):
    logdir: str = tap.arg("--logdir", default=f"laies-lle{LEVEL}-2M", help="Experiment directory, rooted under logs/.")
    n_steps: int = tap.arg("--n-steps", default=2_000_000, help="Number of training steps per run.")
    n_seeds: int = tap.arg("--n-seeds", default=8, help="Number of repetitions (one run per seed).")
    start_seed: int = tap.arg("--start-seed", default=0)
    n_jobs: int = tap.arg("--n-jobs", default=8, help="Number of runs to train in parallel.")
    gpus: list[int] = tap.arg(
        "--gpus",
        nargs="+",
        default=list(range(8)),
        help="GPUs the runs may use. One run needs ~6GB, so the 8 runs are scattered one per GPU.",
    )
    gpu_strategy: Literal["scatter", "group"] = tap.arg("--gpu-strategy", default="scatter")
    test_interval: int = tap.arg("--test-interval", default=50_000)
    n_tests: int = tap.arg("--n-tests", default=10, help="Number of test episodes per evaluation.")
    epsilon_anneal: int = tap.arg(
        "--epsilon-anneal",
        default=50_000,
        help="Steps to anneal epsilon from 1.0 to 0.05. The paper uses 50k, and 500k on hard exploration tasks.",
    )
    beta_idi: float = tap.arg("--beta-idi", default=1.0, help="Weight of the individual diligence reward.")
    beta_cdi: float = tap.arg("--beta-cdi", default=1.0, help="Weight of the collaborative diligence reward.")
    cdi_samples: int = tap.arg("--cdi-samples", default=4, help="Joint counterfactual actions sampled for CDI.")
    quiet: bool = tap.arg("--quiet", default=True)
    resume: bool = tap.arg("--resume", default=False, help="Load an existing experiment and run its missing seeds.")


def external_state_indices(env: LLEConfig):
    """
    Return the flat-state indices of the features that agents should learn to influence.

    The LLE layered state stacks, in this order, one layer per agent position, one laser
    layer per agent colour, then the wall, void, gem and exit layers, flattened
    channel-first. Agent positions and the static map layers (walls, void, exits) are not
    external states: the laser and gem layers are the only ones the agents can alter.

    @ai-generated
    """
    world = env.make_base_env().world
    n_agents = world.n_agents
    layer_size = world.width * world.height
    n_layers = 2 * n_agents + 4
    if env.state_shape[0] != n_layers * layer_size:
        raise ValueError(
            f"Unexpected LLE state shape {env.state_shape} for {n_agents} agents on {world.width}x{world.height}"
        )
    lasers = range(n_agents * layer_size, 2 * n_agents * layer_size)
    gems = range((2 * n_agents + 2) * layer_size, (2 * n_agents + 3) * layer_size)
    return tuple(lasers) + tuple(gems)


def make_trainer(env: LLEConfig, args: Args):
    """
    Build the LAIES trainer with the hyperparameters of the paper.
    """
    qnetwork = qnetworks.from_env(env, recurrent=True, duelling=False, independent=False)
    return LAIES(
        qnetwork=qnetwork,
        mixer=mixers.VDN.from_env(env),
        external_state_indices=external_state_indices(env),
        # QMIX (PyMARL) parameters
        memory_size=5_000,
        batch_size=32,
        train_interval=(1, "episode"),
        train_policy=EpsilonGreedy.linear(1.0, 0.05, args.epsilon_anneal),
        test_policy=ArgMax(),
        target_updater=HardUpdate(200),
        optimiser_type="rmsprop",
        lr=5e-4,
        double_qlearning=True,
        gamma=0.99,
        grad_norm_clipping=10.0,
        # LAIES parameters
        beta_idi=args.beta_idi,
        beta_cdi=args.beta_cdi,
        cdi_samples=args.cdi_samples,
        estm_hidden_size=128,
        estm_lr=3e-4,
        intrinsic_reward_clip=1.0,
    )


def main(args: Args):
    """
    Create (or resume) the experiment and train every seed in parallel.
    """
    seeds = list(range(args.start_seed, args.start_seed + args.n_seeds))
    logpath = Path("logs") / args.logdir
    if args.resume and logpath.exists():
        experiment = marl.Experiment.load(logpath)
        complete = {run.seed for run in experiment.runs if run.is_complete}
        seeds = [seed for seed in seeds if seed not in complete]
        if not seeds:
            LOGGER.info("All %d runs of %s are already complete.", args.n_seeds, logpath)
            return
        LOGGER.info("Resuming %s with the missing seeds %s.", logpath, seeds)
    else:
        env = LLEConfig(LEVEL, obs_type="layered", state_type="flattened")
        trainer = make_trainer(env, args)
        experiment = marl.Experiment.create(env, trainer, logdir=logpath, n_steps=args.n_steps)
        LOGGER.info("Created experiment %s (%s).", experiment.logdir, trainer.name)

    disabled_gpus = [gpu for gpu in range(8) if gpu not in args.gpus]
    experiment.run(
        seeds=seeds,
        n_jobs=args.n_jobs,
        gpu_strategy=args.gpu_strategy,
        disabled_gpus=disabled_gpus,
        test_interval=args.test_interval,
        n_tests=args.n_tests,
        save_weights=True,
        save_actions=True,
        quiet=args.quiet,
        limit_torch_threads=False,
    )


if __name__ == "__main__":
    dotenv.load_dotenv()
    logging.basicConfig(
        handlers=[logging.FileHandler("train_laies_lle6.log", mode="a"), logging.StreamHandler()],
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    try:
        tap.Parser(Args).bind(main).run()
    except KeyboardInterrupt:
        raise
    except Exception as error:
        LOGGER.error("LAIES training command failed: %s", " ".join(sys.argv), exc_info=error)
        raise
