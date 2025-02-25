from lle import LLE


def main():
    env = LLE.level(6).build()
    episode = env.replay()


if __name__ == "__main__":
    pass
