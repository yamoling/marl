import time
import marl
import lle
from rlenv import RLEnv
from lle import WorldState, LLE, Action
from marl.env.lle_shaping import LLEShaping, LLEShapeEachLaser
from marl.env.lle_curriculum import LaserCurriculum
from marl.models import Experiment, SimpleRunner, RLAlgo
from itertools import permutations


if __name__ == "__main__":
    env = LLEShapeEachLaser(LLE.level(6), 0.5, True)
    print(env)
    env.reset()
    env.render("human")
    actions = [
        [Action.STAY.value, Action.STAY.value, Action.SOUTH.value, Action.STAY.value],
        [Action.STAY.value, Action.STAY.value, Action.WEST.value, Action.STAY.value],
        [Action.STAY.value, Action.STAY.value, Action.WEST.value, Action.STAY.value],
        [Action.STAY.value, Action.STAY.value, Action.WEST.value, Action.STAY.value],
        [Action.STAY.value, Action.STAY.value, Action.WEST.value, Action.STAY.value],
        [Action.STAY.value, Action.STAY.value, Action.WEST.value, Action.STAY.value],
        [Action.STAY.value, Action.STAY.value, Action.WEST.value, Action.STAY.value],
        [Action.STAY.value, Action.STAY.value, Action.NORTH.value, Action.STAY.value],
        [Action.STAY.value, Action.STAY.value, Action.SOUTH.value, Action.STAY.value],
        [Action.STAY.value, Action.STAY.value, Action.SOUTH.value, Action.STAY.value],
        [Action.STAY.value, Action.STAY.value, Action.EAST.value, Action.STAY.value],
    ]
    actions2 = [
        [Action.SOUTH.value] * 4,
        [Action.SOUTH.value] * 4,
        [Action.SOUTH.value] * 4,
        [Action.SOUTH.value] * 3 + [Action.STAY.value],
        [Action.STAY.value, Action.SOUTH.value, Action.SOUTH.value, Action.WEST.value],
        [Action.WEST.value, Action.SOUTH.value, Action.STAY.value, Action.SOUTH.value],
        [Action.STAY.value, Action.EAST.value, Action.WEST.value, Action.STAY.value],
        [Action.SOUTH.value, Action.EAST.value, Action.SOUTH.value, Action.SOUTH.value],
        [Action.SOUTH.value, Action.EAST.value] + [Action.SOUTH.value] * 2,
        [Action.SOUTH.value, Action.EAST.value] + [Action.SOUTH.value] * 2,
        [Action.SOUTH.value] * 4,
    ]
    score = 0
    for action in actions:
        obs, reward, done, truncated, info = env.step(action)
        score += reward
        print(reward, score)
        env.render("human")
        input("Press Enter to continue...")
