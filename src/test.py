from typing import Any
from numpy import int32, ndarray
from run import Arguments, main as run_main
from lle import LLE, Action
import marl
import numpy as np
import polars as pl
import rlenv
import random
from marl.training import DQNTrainer, SoftUpdate
from marl.env.wrappers.random_initial_pos import RandomInitialPos

from marl.env.wrappers.prevent_actions import PreventActions

def get_action():
    action = input("Action: ")
    match action.strip().lower():
        case "n" | "z":
            return Action.NORTH
        case "s":
            return Action.SOUTH
        case "e" | "d":
            return Action.EAST
        case "w" | "q":
            return Action.WEST
        case other:
            return None


def play(env: LLE):
    print("New game")
    env.reset()
    env.render("human")
    print("Available:", env.available_actions())
    action = get_action()
    while action:
        action = [action.value]
        _, _, done, _, _ = env.step(action)
        print("Available:", env.available_actions())
        env.render("human")
        if done:
            break
        action = get_action()

from marl.utils import RandomAlgo
from marl.training import NoTrain

def exit_rate():
    files= [
        "logs/exploration-bw-0/run_2024-04-19_16:10:05.216887_seed=0/train.csv",
        "logs/exploration-bw-1/run_2024-04-19_16:10:15.270965_seed=0/train.csv",
        "logs/exploration-bw-2/run_2024-04-19_16:10:25.463659_seed=0/train.csv",
        "logs/exploration-bw-3/run_2024-04-19_16:10:36.094510_seed=0/train.csv",
        "logs/exploration-bw-4/run_2024-04-19_16:10:46.659614_seed=0/train.csv",
        "logs/exploration-bw-5/run_2024-04-19_16:10:56.694036_seed=0/train.csv"
    ]
    for f in files:
        df = pl.read_csv(f)
        x = df.mean()
        print(x)

if __name__ == "__main__":
    # exit_rate()
    # exit()
    filenames = []
    time_limit = 14
    for width in range(6):
        env = rlenv.Builder(PreventActions(width)).time_limit(time_limit).build()
        # input("Press enter to start")
        # img = env.render("rgb_array")
        # import cv2
        # cv2.imwrite("exploration.png", img)
        # exit(0)
        algo = RandomAlgo(env)
        trainer = NoTrain()
        exp = marl.Experiment.create(f"logs/exploration-bw-{width}-{time_limit}", algo, trainer, env, 200_000, 0)
        exp.run(0, "conservative", 0, n_tests=0)
        run = list(exp.runs)[0]
        filenames.append(run.train_filename)
    for f in filenames:
        print(f)
        df = pl.read_csv(f).mean()
        print(df)
        
    
    