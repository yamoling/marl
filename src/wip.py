from laser_env import LaserEnv
import rlenv
import marl
from marl.debugging import QLearningInspector


env, test_env = rlenv.Builder(LaserEnv("maps/lvl3"))\
        .agent_id()\
        .extrinsic_reward("exp", anneal=50, initial_reward=1)\
        .build_all()
    
algo = marl.RecurrentVDN(env, test_env=test_env, train_policy=marl.policy.DecreasingEpsilonGreedy(env.n_agents, decrease_amount=5e-5), log_path="debug")

app = QLearningInspector(algo, debug=True)
print("Done")
