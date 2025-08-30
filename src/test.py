import marl
import marlenv

if __name__ == "__main__":
    env = marlenv.catalog.DeepSea(25)
    env = marl.env.StateCounter(env)
