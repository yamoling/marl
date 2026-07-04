import os

import lle
from lle.generator import WorldFilter


def main():
    for world in lle.generate("random", n_agents=3, n_lasers=2, n=1000, cooperative=True):
        pass
    worlds = lle.generate(
        kind="random",
        max_attempts=50_000_000,
        n=1_000_000,
        filter=WorldFilter.cooperative(50, t_min=8),
        n_lasers=n_lasers,
        n_agents=n_agents,
        n_jobs=30,
    )
    directory = f"maps/{n_agents}/cooperative-8-50-{n_lasers}-lasers"
    os.makedirs(directory, exist_ok=True)
    n_existing = len(os.listdir(directory))
    for i, world in enumerate(worlds):
        i = i + n_existing
        with open(f"{directory}/{i}", "w") as f:
            f.write(world.world_string)


if __name__ == "__main__":
    main()
