import os
from pathlib import Path

import lle
from lle import CooperationLevel
from lle.generator import try_generate

if __name__ == "__main__":
    for generator in ("constructive", "random", "level6_style"):
        for cooperation in lle.CooperationLevel:
            if cooperation in lle.CooperationLevel.COOPERATIVE:
                continue
            # if generator == "constructive" and cooperation in (
            #     CooperationLevel.CHAIN,
            #     CooperationLevel.FULLY_COUPLED,
            # ):
            #     continue
            if generator == "random" and cooperation in (CooperationLevel.FULLY_COUPLED,):
                continue
            if generator == "level6_style" and not cooperation.is_cooperative:
                continue
            path = Path("maps/pool/", generator, cooperation.name)
            path.mkdir(exist_ok=True, parents=True)
            print("Generating", path)
            n_existing = len(os.listdir(path))
            to_generate = 1000 - n_existing
            if to_generate == 0:
                continue
            worlds = try_generate(
                generator,
                max_attempts=100_000,
                width=13,
                height=12,
                n=to_generate,
                t_max=39,
                cooperation=cooperation,
                n_agents=4,
                n_lasers=3,
                n_jobs=8,
            )
            for i, world in enumerate(worlds):
                with open(path / f"{n_existing + i}", "w") as f:
                    f.write(world.world_string)
