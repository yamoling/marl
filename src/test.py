import time
import marl
import lle
from rlenv import RLEnv
from lle import WorldState, LLE, Action
from marl.env.lle_shaping import LLEShaping
from marl.env.lle_curriculum import LaserCurriculum
from marl.models import Experiment, SimpleRunner, RLAlgo
from itertools import permutations


if __name__ == "__main__":
    exp = Experiment.load("logs/curriculum-lasers-test-lvl6")
    env = Experiment.load("logs/curriculum-lasers").test_env
    exp.test_on_other_env(env, "logs/test", 10)
