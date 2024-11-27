from lle import LLE
import marlenv
from time import time
from marl.algo import mcts

env = LLE.level(1).single_objective()
env = marlenv.Builder(env).time_limit(40).build()
env.seed(0)
obs = env.reset()

done = truncated = False
while not done and not truncated:
    env.render("human")
    start = time()
    action = mcts.search(env, time_limit_ms=1000)
    print(action)
    print(f"Search duration: {time() - start}")
    _, _, done, truncated, _ = env.step(action)
    print("trucated", truncated, "done", done)

env.render("human")
input("Press Enter to continue...")
