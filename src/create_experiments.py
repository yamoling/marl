import rlenv
import marl
from laser_env import ObservationType, StaticLaserEnv
from marl.models import Experiment


def create_experiments():
    marl.seed(0)
    batch_size = 64
    memory_size = 50_000
    level = "maps/normal/lvl6"
    n_steps = 1_500_000
    for alpha in [0.5, 0.6, 0.7]:
        for beta in [0.3, 0.4, 0.5]:
            env = StaticLaserEnv(level, ObservationType.LAYERED)
            time_limit = int(env.width * env.height / 2)
            current_env, test_env = rlenv.Builder(env).agent_id().time_limit(time_limit).build_all()
            
            # E-greedy decreasing from 1 to 0.05 over 400_000 steps
            min_eps = 0.05
            decrease_amount = (1 - min_eps) / 400_000
            train_policy = marl.policy.DecreasingEpsilonGreedy(1, decrease_amount, min_eps)
            test_policy = marl.policy.ArgMax()

            qnetwork = marl.nn.model_bank.CNN.from_env(current_env)
            mixer = marl.qlearning.mixers.VDN(env.n_agents)
            # mixer = marl.qlearning.mixers.QMix(env.state_shape[0], env.n_agents, 64)
            memory = marl.models.TransitionMemory(memory_size)
            memory = marl.models.PrioritizedMemory(memory, beta_anneal_steps=n_steps, alpha=alpha, beta=beta)

            algo = marl.qlearning.MixedDQN(
                qnetwork=qnetwork,
                batch_size=batch_size,
                train_policy=train_policy,
                test_policy=test_policy,
                gamma=0.95,
                memory=memory,
                mixer=mixer
            )

            name = level
            if level.startswith("maps/"):
                name = level[5:11] + level[-1]
            logdir = f"logs/{name}-{mixer.name}-{memory.__class__.__name__}-alpha_{alpha}-beta_{beta}"
            # logdir = "test-mixed-vdn-double-qlearning"
            exp = Experiment.create(logdir, algo=algo, env=current_env, test_interval=5000, test_env=test_env)
            # exp.create_runner().train(1)
            print("Created experiment:", exp.logdir)


if __name__ == "__main__":
    create_experiments()
