import marl
from lle import LLE
import marlenv


def main():
    env = LLE.level(3).single_objective()
    env = marlenv.Builder(env).time_limit(78).build()

    trainer = marl.algo.intrinsic_reward.IndividualLocalGraphTrainer(env)  # type: ignore
    runner = marl.Runner(env, trainer=trainer)
    runner.run("logs/tests", n_steps=50_000)


if __name__ == "__main__":
    main()
