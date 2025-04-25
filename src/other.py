from lle import LLE, Action


def main():
    env = LLE.from_file("maps/pool/world-31").build()
    env.reset()
    actions = [Action.STAY, Action.SOUTH, Action.SOUTH, Action.STAY]
    env.step([a.value for a in actions])


if __name__ == "__main__":
    main()
