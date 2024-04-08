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


def get_memory_usage_by_pid(pid: int):
    command = ["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_name", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        raise ValueError(f"Failed to get memory usage: {result.stderr}")
    total = 0
    for line in result.stdout.decode().split("\n"):
        print(line)
        if line.strip() == "":
            continue
        key, usage, name = line.split(",")
        if int(key) == pid:
            total += int(usage)
    return total


if __name__ == "__main__":
    print(get_memory_usage_by_pid(1644092))
    exit()

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
