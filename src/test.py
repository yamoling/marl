import time
import marl
import lle
from rlenv import RLEnv, Builder
from lle import WorldState, LLE, Action
from marl.env.lle_shaping import LLEShaping, LLEShapeEachLaser
from marl.env.lle_curriculum import LaserCurriculum
from marl.models import Experiment, SimpleRunner, RLAlgo, Run
from marl.qlearning import DQN
from itertools import permutations


if __name__ == "__main__":
    env = LLE.level(6)
    env = Builder(env).agent_id().time_limit(78, add_extra=True).build()
    exp = Experiment.load("logs/vdn-baseline")
    run = Run.load("logs/vdn-baseline/run_2024-03-26_14:24:27.140843_seed=4")

    algo = exp.algo
    assert isinstance(algo, DQN)
    algo.load(run.get_saved_algo_dir(1_000_000))

    values = []
    algo.set_testing()
    actions = [
        [Action.SOUTH.value] * 4,
        [Action.SOUTH.value] * 4,
        [Action.SOUTH.value] * 4,
        [Action.SOUTH.value] * 4,
    ]

    obs = env.reset()
    for action in actions:
        env.step(action)
