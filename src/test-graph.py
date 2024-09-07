from marl.other.subgoal import LocalGraph
import random
from icecream import ic

lg = LocalGraph[int]()
trajectory = [random.randint(0, 2) for _ in range(50)] + [2] + [random.randint(3, 5) for _ in range(50)]

lg.add_trajectory(trajectory)
score, cut_set, node = lg.partition()

ic(score, cut_set, node)
