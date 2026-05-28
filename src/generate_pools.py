from pathlib import Path

import lle

if __name__ == "__main__":
    for generator in ("constructive", "level6_style", "random"):
        path = Path("maps/pool/") / generator
        path.mkdir(exist_ok=True)
        for i, world in enumerate(lle.generate(generator, width=13, height=12, n=1000, t_max=39)):
            print(world.world_string)
            with open(path / f"{i}", "w") as f:
                f.write(world.world_string)
