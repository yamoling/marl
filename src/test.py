import marl
import marlenv
from lle import LLE


def main():
    env = LLE.level(2).obs_type("state").build()
    env = marlenv.Builder(env).agent_id().time_limit(50, add_extra=False).build()
    trainer = marl.training.QLearning(env.n_actions, env.n_agents)
    exp = marl.Experiment.create(env, 1_000_000, trainer=trainer)
    exp.run()


if __name__ == "__main__":
    main()
