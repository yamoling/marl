from lle import World, LLE, exceptions
import random
import time
import os


class Generator:
    def __init__(
        self,
        width=10,
        height=10,
        num_agents=2,
        num_gems=0,
        num_lasers=0,
        wall_density=0.2,
    ):
        self._width = width
        self._height = height
        self._num_agents = num_agents
        self._num_gems = num_gems
        self._num_lasers = num_lasers
        self._wall_density = wall_density

        assert 1 <= num_agents <= 4, "Number of agents must be between 1 and 4"
        if wall_density > self.max_density:
            raise ValueError(f"Wall density {wall_density} is too high. Max density is {self.max_density}")

    def generate(self):
        """
        Generate a random world with the given parameters.

        Raises
            - `lle.exceptions.ParsingError` if the generated world is invalid, for instance
            if a laser is beaming trough an agent's start tile.
        """
        assert self.num_tiles * (1 - self._wall_density) >= 2 * self._num_agents + self._num_gems + self._num_lasers, (
            "Not enough tiles to place all the entities"
        )
        empty_tiles = [(x, y) for x in range(self._width) for y in range(self._height)]
        grid = [["." for _ in range(self._width)] for _ in range(self._height)]
        for i in range(self._num_agents):
            self._place_randomly(grid, f"S{i}", empty_tiles)
            self._place_randomly(grid, "X", empty_tiles)
        for i in range(self._num_gems):
            self._place_randomly(grid, "G", empty_tiles)
        for i in range(self._num_lasers):
            direction = random.choice("NSEW")
            agent = random.randint(0, self._num_agents - 1)
            self._place_randomly(grid, f"L{agent}{direction}", empty_tiles)
        self._place_walls(grid, empty_tiles)

        str_grid = "\n".join(" ".join(row) for row in grid)
        return World(str_grid)

    @property
    def num_tiles(self) -> int:
        return self._width * self._height

    @property
    def max_density(self) -> float:
        num_tiles = self.num_tiles
        return (num_tiles - self._num_agents * 2 - self._num_gems) / num_tiles

    def _place_randomly(self, grid: list[list[str]], tile: str, empty_tiles: list[tuple[int, int]]):
        i = random.randint(0, len(empty_tiles) - 1)
        x, y = empty_tiles.pop(i)
        grid[y][x] = tile

    def _place_walls(self, grid: list[list[str]], empty_tiles: list[tuple[int, int]]):
        num_walls = int(self._wall_density * self.num_tiles) - self._num_lasers
        while num_walls > 0:
            self._place_randomly(grid, "@", empty_tiles)
            num_walls -= 1

    def seed(self, seed_value: int):
        random.seed(seed_value)


def main():
    lvl6 = LLE.level(6).build()
    generator = Generator(num_agents=4, num_lasers=2, num_gems=lvl6.world.n_gems, wall_density=0.05, width=lvl6.width, height=lvl6.height)
    n_saved = len([f for f in os.listdir(".") if f.startswith("world-")])
    print(n_saved)
    while True:
        try:
            world = generator.generate()
            env = LLE(world)
            env.render()
            time.sleep(0.1)
            env.render()
            time.sleep(0.1)
            do_save = input("Save? (y/n): ")
            match do_save:
                case "y" | "Y" | "":
                    s = world.world_string
                    with open(f"world-{n_saved}", "w") as f:
                        f.write(s)
                    n_saved += 1
                    print("Saved")
                case _:
                    print("Not saved")
        except exceptions.ParsingError as e:
            print("ParsingError:", e)


def remove_gems():
    for file in os.listdir("."):
        if not file.startswith("world-"):
            continue
        with open(file, "r") as f:
            content = f.read()
        n_gems = content.count("G")
        if n_gems == 5:
            content = content.replace("G", ".", count=1)
            print(f"Removed a gem from {file}")
        with open(file, "w") as f:
            f.write(content)


if __name__ == "__main__":
    main()
