import rlenv
import marl

if __name__ == "__main__":
    print(marl.__version__)
    env = rlenv.make("smac:3m")
    dqn = marl.VDN(env, train_policy=marl.policy.DecreasingEpsilonGreedy(env.n_agents, decrease_amount=5e-5))
    dqn.train(n_steps=200_000, test_interval=5000)