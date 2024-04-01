import marl
import lle
from lle import WorldState, LLE, Action
from marl.env.lle_shaping import LLEShaping
from itertools import permutations


env = LLE.level(3, obs_type=lle.ObservationType.LAYERED_PADDED)
env = LLEShaping(env, 1.0)
env.reset()
env.render("human")
r = env.step([Action.SOUTH.value, Action.SOUTH.value])[1]
print(r)
env.render("human")
input("Press Enter to continue...")
r = env.step([Action.SOUTH.value, Action.SOUTH.value])[1]
print(r)
env.render("human")
input("Press Enter to continue...")
r = env.step([Action.SOUTH.value, Action.SOUTH.value])[1]
print(r)
env.render("human")
input("Press Enter to continue...")
r = env.step([Action.SOUTH.value, Action.SOUTH.value])[1]
print(r)
env.render("human")
input("Press Enter to continue...")
r = env.step([Action.SOUTH.value, Action.SOUTH.value])[1]
print(r)
env.render("human")
input("Press Enter to continue...")
r = env.step([Action.SOUTH.value, Action.SOUTH.value])[1]
print(r)
env.render("human")
input("Press Enter to continue...")
r = env.step([Action.SOUTH.value, Action.SOUTH.value])[1]
print(r)
env.render("human")
input("Press Enter to continue...")
r = env.step([Action.SOUTH.value, Action.SOUTH.value])[1]
print(r)
