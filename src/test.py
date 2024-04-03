import time
import marl
import lle
from lle import WorldState, LLE, Action
from marl.env.lle_shaping import LLEShaping
from marl.env.lle_curriculum import LaserCurriculum
from itertools import permutations


env = LLE.level(6, obs_type=lle.ObservationType.LAYERED_PADDED)
env = LaserCurriculum(env)
env.reset()

env.render("human")
time.sleep(0.2)
for t in range(0, 1_000_000, 100_000):
    r = ""
    while r != "n":
        env.t = t
        env.reset()
        env.render("human")
        r = input(f"'n' to continue to next step, any other key to generate a new environment for t={t}: ")
