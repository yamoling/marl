"""Render a rollout on the lift/button map to a GIF for visual inspection.

Three policies are available via --policy:
  - "random": no training required, just wanders the map.
  - "trained": loads a trained agent's latest checkpoint via --logdir/--seed.

Usage:
    uv run python examples/render_lift_button.py --policy random
    uv run python examples/render_lift_button.py --policy trained --logdir logs/VDN-QRNN-double-duelling-LLE-lift.toml --seed 0
"""

import argparse

from lle import Action
from PIL import Image

from marl import Experiment, algos
from marl.env import LLEConfig
from marl.runners import seeded_rollout

# MAP_PATH = 2
MAP_PATH = "./lift2.toml"

OUTPUT_PATH = "lift_button_rollout.gif"

N, S, E, W, STAY, TRIGGER = (
    Action.NORTH,
    Action.SOUTH,
    Action.EAST,
    Action.WEST,
    Action.STAY,
    Action.TRIGGER,
)


def load_trained_agent(logdir: str, seed: int | None):
    experiment = Experiment.load(logdir)
    run = experiment.get_run(seed) if seed is not None else next(iter(experiment.runs))
    assert run is not None, f"No run found for seed={seed} in {logdir}"
    full_run = run.to_full()
    agent = full_run.make_agent()

    test_root = full_run.runpath / "test"
    checkpoints = [
        int(path.name) for path in test_root.iterdir() if path.is_dir() and path.name.isdigit() and any(path.iterdir())
    ]
    if not checkpoints:
        raise FileNotFoundError(f"No saved checkpoints found in {test_root}")
    checkpoint_step = max(checkpoints)
    agent.load(test_root / str(checkpoint_step))
    print(f"Loaded checkpoint at step {checkpoint_step} from {full_run.runpath}")
    return agent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("solution", "random", "trained"), default="solution")
    parser.add_argument("--logdir", help="Experiment logdir to load a trained agent from (--policy trained)")
    parser.add_argument("--seed", type=int, default=None, help="Run seed to load within --logdir")
    parser.add_argument("--map", default=MAP_PATH, help="Path to the .toml map")
    parser.add_argument("--out", default=OUTPUT_PATH, help="Output GIF path")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--time-limit", type=int, default=30, help="Episode time limit in steps")
    args = parser.parse_args()

    # lift.toml's auto time limit (width * height // 2) is only 4 steps, too short to
    # show the lift/button mechanic (the hand-solved path takes 10 steps) -- give it room.
    env = LLEConfig(args.map, obs_type="flattened", time_limit=args.time_limit).make()

    if args.policy == "trained":
        if not args.logdir:
            parser.error("--policy trained requires --logdir")
        agent = load_trained_agent(args.logdir, args.seed)
    else:
        agent = algos.NoTrain.discrete(env).make_agent()
    episode, frames, _ = seeded_rollout(env, agent, seed=0, compute_frames=True)
    print(f"Episode finished in {len(episode)} steps, metrics={episode.metrics}")

    images = [Image.fromarray(frame) for frame in frames]
    duration_ms = int(1000 / args.fps)
    images[0].save(args.out, save_all=True, append_images=images[1:], duration=duration_ms, loop=0)
    print(f"Saved {len(images)} frames to {args.out}")


if __name__ == "__main__":
    main()
