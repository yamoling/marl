import time
import marl
import lle
from rlenv import RLEnv, Builder
from lle import WorldState, LLE, Action
from marl.env.lle_shaping import LLEShaping, LLEShapeEachLaser
from marl.env.lle_curriculum import LaserCurriculum
from marl.models import Experiment, SimpleRunner, RLAlgo, Run
from marl.qlearning import DQN
import subprocess
from itertools import permutations
from marl.env.b_shaping import BShaping

if __name__ == "__main__":
    # print(get_memory_usage_by_pid(1644092))
    env = LLE.from_str(
        """
 .  S0 S1 . 
 .  .  .  .   
L0E .  .  .   
 .  .  .  .   
 .  .  .  .
 .  .  .  .   
 .  .  .  .
 .  .  .  .   
 .  .  .  .
 .  .  .  .   
 .  .  .  .   
 .  X  X  . 
"""
    ).build()
    # env = LLE.from_file("maps/1b").build()
    env = BShaping(env, env.world, 1.0, 2)
    obs = env.reset()
    env.render("human")
    time.sleep(0.5)
    env.render("human")

    for i in range(10):
        print(obs.extras[0])
        input()
        obs, r, *_ = env.step([Action.SOUTH.value] * env.n_agents)
        print(r)
        env.render("human")
    input()
