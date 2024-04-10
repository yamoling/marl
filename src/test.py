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


from marl.env.b_shaping import BShaping

if __name__ == "__main__":
    # print(get_memory_usage_by_pid(1644092))
    env = LLE.from_file("maps/1b").build()
    env = BShaping(env, 1.0, 0)
