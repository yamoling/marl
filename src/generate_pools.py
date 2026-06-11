import os
from pathlib import Path

import lle
from lle import CooperationLevel

if __name__ == "__main__":
    worlds = lle.generate(
        kind="random",
        max_attempts=100_000,
        width=13,
        height=12,
        n=1000,
        t_max=39,
        cooperative=True,
        n_agents=4,
    )
    for i, world in enumerate(worlds):
        os.makedirs("maps/pool2/random/distributed", exist_ok=True)
        with open(f"maps/pool2/random/distributed/{i}", "w") as f:
            f.write(world.world_string)
    exit()
    for generator in ("constructive", "random", "level6_style"):
        for cooperation in lle.CooperationLevel:
            if cooperation in lle.CooperationLevel.COOPERATIVE:
                continue
            if generator == "constructive" and cooperation in (CooperationLevel.CHAIN,):
                continue
            if generator == "random" and cooperation in (CooperationLevel.FULLY_COUPLED,):
                continue
            if generator == "level6_style" and not cooperation.is_cooperative:
                continue
            path = Path("maps/pool2/", generator, cooperation.name)
            path.mkdir(exist_ok=True, parents=True)
            print("Generating", path)
            n_existing = len(os.listdir(path))
            to_generate = 1000 - n_existing
            if to_generate == 0:
                continue
            n_agents = 4
            if cooperation == CooperationLevel.FULLY_COUPLED:
                n_lasers = n_agents
            else:
                n_lasers = 3
            worlds = lle.generate(
                "constructive",
                max_attempts=100_000,
                width=13,
                height=12,
                n=to_generate,
                t_max=39,
                cooperation=cooperation,
                n_agents=n_agents,
                n_lasers=n_lasers,
                n_jobs=8,
            )
            for i, world in enumerate(worlds):
                with open(path / f"{n_existing + i}", "w") as f:
                    f.write(world.world_string)
