import rlenv
import marl
from laser_env import Difficulty, ObservationType, StaticLaserEnv
from marl.utils.env_pool import pool_from_zip
from marl.models import Experiment



def create_env_pool_experiments():
    for difficulty in [Difficulty.MEDIUM, Difficulty.HARD]:
        for batch_size in [64, 128]:
            for memory_size in [50_000, 100_000]:
                map_size = 5
                time_limit = map_size * map_size
                

                current_env, test_env = pool_from_zip(f"maps/{map_size}x{map_size}.zip", difficulty, ObservationType.FLATTENED)
                current_env = rlenv.Builder(current_env).agent_id().time_limit(time_limit).build()
                test_env = rlenv.Builder(test_env).agent_id().time_limit(time_limit).build()
                
                
                # E-greedy decreasing from 1 to 0.05 over 600000 steps
                min_eps = 0.05
                decrease_amount = (1 - min_eps) / 600_000
                algo = marl.utils.RandomAgent(current_env.n_actions, current_env.n_agents)
                train_policy = marl.policy.DecreasingEpsilonGreedy(1, decrease_amount, min_eps)
                test_policy = marl.policy.ArgMax()

                qnetwork = marl.nn.model_bank.MLP.from_env(current_env)
                memory = marl.models.TransitionMemory(memory_size)
                # memory = marl.models.PrioritizedMemory(memory)
                
                algo = (marl.DeepQBuilder()
                        .qnetwork(qnetwork)
                        .batch_size(batch_size)
                        .train_policy(train_policy)
                        .test_policy(test_policy)
                        .gamma(0.95)
                        .memory(memory)
                        # .vdn()
                        .build())

                logdir = f"logs/{algo.name}-{map_size}x{map_size}-{difficulty.name}-batch_{batch_size}-{memory.__class__.__name__}_{memory_size}"
                # logdir = "test"

                experiment = Experiment.create(logdir, algo=algo, env=current_env, n_steps=1_000_000, test_interval=5000, test_env=test_env)




def create_static_experiments():
    batch_size = 64
    memory_size = 50_000
    for level in ["lvl5", "lvl6"]:
        for prioritized in [True, False]:
            env = StaticLaserEnv(level, ObservationType.LAYERED)
            time_limit = env.width * env.height
            current_env, test_env = rlenv.Builder(env).agent_id().time_limit(time_limit).build_all()
            
            # E-greedy decreasing from 1 to 0.05 over 600000 steps
            min_eps = 0.05
            decrease_amount = (1 - min_eps) / 600_000
            algo = marl.utils.RandomAgent(current_env.n_actions, current_env.n_agents)
            train_policy = marl.policy.DecreasingEpsilonGreedy(1, decrease_amount, min_eps)
            test_policy = marl.policy.ArgMax()

            # qnetwork = marl.nn.model_bank.MLP.from_env(current_env)
            qnetwork = marl.nn.model_bank.CNN.from_env(current_env)
            memory = marl.models.TransitionMemory(memory_size)
            if prioritized:
                memory = marl.models.PrioritizedMemory(memory)
            
            algo = (marl.DeepQBuilder()
                    .qnetwork(qnetwork)
                    .batch_size(batch_size)
                    .train_policy(train_policy)
                    .test_policy(test_policy)
                    .gamma(0.95)
                    .memory(memory)
                    .vdn()
                    .build())

            logdir = f"logs/{current_env.name}-{level}-{algo.name}-{qnetwork.name}-{memory.__class__.__name__}_{memory_size}"
            # logdir = "test"
            print("Creating experiment:", logdir)
            Experiment.create(logdir, algo=algo, env=current_env, n_steps=2_000_000, test_interval=5000, test_env=test_env)



if __name__ == "__main__":
    create_static_experiments()
