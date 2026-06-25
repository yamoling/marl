import os

import lle
from lle.generator import WorldFilter


def main():
    for n_agents in [3, 4]:
        for delta in [-1, 0]:
            n_lasers = n_agents + delta
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
    # worlds = lle.generate(
    #     max_attempts=10_000_000,
    #     n=100_000,
    #     filter=WorldFilter.cooperative(40, t_min=8),
    #     n_lasers=3,
    #     n_agents=3,
    #     n_jobs=20,
    # )
    # directory = "maps/pool2/cooperative-8-40"
    # os.makedirs(directory, exist_ok=True)
    # for i, world in enumerate(worlds):
    #     with open(f"{directory}/{i}", "w") as f:
    #         f.write(world.world_string)
