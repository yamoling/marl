"""Build an LLEPool from the 10k canonical asymmetric layouts.

If the pool fails to build because some layouts don't share the same shape
as the rest, scan every layout and report which ones differ, how, and how many.
"""

import logging
import os
from collections import Counter
from pathlib import Path

from lle import World

from marl.env import LLEPool

POOL_DIR = Path("layouts/canonical/asymmetric")


def scan_shapes(pool_dir: Path) -> dict[tuple[int, int, int, int], list[str]]:
    """
    Group every layout file in `pool_dir` by (width, height, n_agents, n_lasers).

    @ai-generated
    """
    files = sorted(os.listdir(pool_dir))
    groups: dict[tuple[int, int, int, int], list[str]] = {}
    for f in files:
        world = World.from_file(str(pool_dir / f))
        key = (world.width, world.height, world.n_agents, len(world.laser_sources))
        groups.setdefault(key, []).append(f)
    return groups


def report_mismatches(groups: dict[tuple[int, int, int, int], list[str]]):
    """
    Print a summary of layout shapes and flag every group that isn't the majority shape.

    @ai-generated
    """
    majority_key = max(groups, key=lambda k: len(groups[k]))
    print(f"Found {len(groups)} distinct layout shape(s):")
    for key, files in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        width, height, n_agents, n_lasers = key
        flag = "" if key == majority_key else "  <-- MISMATCH"
        print(f"  {width}x{height}, {n_agents} agents, {n_lasers} laser(s): {len(files)} layouts{flag}")
    for key, files in groups.items():
        if key == majority_key:
            continue
        print(f"  Mismatched files for {key}: {files[0]} .. {files[-1]} ({len(files)} total)")
    return majority_key


def build_pool(pool_dir: Path, pool_size: int, time_limit: int):
    """
    Build an LLEPool from `pool_dir` and return the underlying env, or None if construction fails.

    @ai-generated
    """
    pool = LLEPool(pool_dir, pool_size, time_limit=time_limit, state_type="flattened")
    try:
        env = pool.env
        logging.info("Pool built successfully with %d layouts.", pool_size)
        return env
    except AssertionError as e:
        logging.error("Pool construction failed: %s", e)
        return None


def main():
    """
    @ai-generated
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    n_layouts = len(os.listdir(POOL_DIR))
    first_world = World.from_file(str(POOL_DIR / sorted(os.listdir(POOL_DIR))[0]))
    time_limit = first_world.width * first_world.height

    env = build_pool(POOL_DIR, n_layouts, time_limit)
    if env is not None:
        return

    print("\nScanning all layouts to find the mismatch...\n")
    groups = scan_shapes(POOL_DIR)
    majority_key = report_mismatches(groups)
    n_mismatched = sum(len(files) for key, files in groups.items() if key != majority_key)
    print(f"\n{n_mismatched}/{n_layouts} layouts don't match the majority shape {majority_key}.")


if __name__ == "__main__":
    main()
