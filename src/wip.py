from laser_env import LaserEnv
import rlenv
import marl
from marl.debugging.server import run


run(port=5174, debug=True)
exit(0)

env, test_env = rlenv.Builder(LaserEnv("maps/lvl3"))\
        .agent_id()\
        .record(folder="files/test/videos/")\
        .extrinsic_reward("exp", anneal=50, initial_reward=1)\
        .build_all()
    

algo = marl.VanillaQLearning(env, test_env, log_path="debug")
algo = marl.debugging.FileWrapper(algo, directory="files/")
algo.train(n_steps=40_000, test_interval=2500, n_tests=2)

# debugger = QLearningInspector(algo)
# debugger.run(debug=True)

print("Done")
