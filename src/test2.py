from lle import LLE
import time


def main():
    env = LLE.from_file("maps/lvl6-start-above.toml").build()
    while True:
        env.reset()
        env.render()
        time.sleep(0.2)
        env.render()
        time.sleep(0.2)


if __name__ == "__main__":
    main()
