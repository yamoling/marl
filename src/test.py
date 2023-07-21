import rlenv
import marl
from lle import LLE, ObservationType


def main():
    env = LLE.from_file("lvl6", ObservationType.LAYERED)
    env.reset()
    env.render("rgb")
    input()


if __name__ == "__main__":
    main()
