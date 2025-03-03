from lle import LLE
import time
from marl.env.wrappers.randomized_lasers import RandomizedLasers


def main():
    env = RandomizedLasers(LLE.from_file("maps/lvl6-start-above.toml").build())  # , [(4, 0), (6, 12)])
    while True:
        env.reset()
        env.render()
        time.sleep(0.2)
        env.render()
        time.sleep(0.2)


if __name__ == "__main__":
    main()
