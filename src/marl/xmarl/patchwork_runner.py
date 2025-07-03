from marl.models import Experiment
from marlenv.models import Episode

import pathlib
import os

from marl.xmarl.distilers.sdt import SoftDecisionTree
from marl.xmarl import FrameViewer, HeatmapXFrameViewer

from marl.xmarl.distilers.utils import get_env_infos

LOG_PATH = pathlib.Path("logs")

def get_selection(cur_dir: str, file_list: str) -> pathlib.Path:
    print("Select one of the following files (you may use the list index):")
    print(file_list)
    file_str = input()
    if file_str in  file_list:
        file_path = cur_dir / file_str
    elif file_str.lstrip("-").isnumeric() and abs(int(file_str)) < len(file_list):
        file_str = file_list[int(file_str)]
        file_path = cur_dir / file_str
    else:
        raise Exception("Not a valid file")
    return file_path

def handle_selection() -> tuple[Experiment, pathlib.Path, pathlib.Path]:
    exp_list = os.listdir(LOG_PATH)

    exp_path = get_selection(LOG_PATH, exp_list)     
    experiment = Experiment.load(exp_path)
    if experiment.logdir != exp_path:
            experiment.logdir = exp_path
    print()

    exp_cont = os.listdir(exp_path)
    run_list = [run for run in exp_cont if "run" in run]
    run_path = get_selection(exp_path, run_list)
    print()

    test_path = run_path / "test"
    test_list = os.listdir(test_path)
    timestep_path = get_selection(test_path, test_list)
    print()

    distil_path = None
    if "distil" in exp_cont:
        distil_list = [distil for distil in os.listdir(exp_path/"distil") if "distil" in distil]
        if len(distil_list) > 0:
            print("Distillation(s) of the experiment's model found, do you want to visualize it? \n[Y/N]")
            yn = input()
            if str.upper(yn) == "Y": distil_path = get_selection(exp_path/"distil", distil_list)

    return experiment, timestep_path, distil_path

def handle_distillation(episode: Episode, distil_path: pathlib.Path):
    if "sdt" in str(distil_path):
        distiller = SoftDecisionTree.load(distil_path)
        print("SDT has two types of explanations to provide, choose by inputting the index: \n0: Forward (Traverse greediest path to action) - 1: Backward (Filters of path to original action)")
        e_type = input() == "1"
        distilled_filters, distilled_extras, distilled_actions, agent_pos = distiller.distil_episode(episode, e_type) # Give shape for obs and extras to be computed there, or reshape after, need to separate extras that's annoying
    else: raise Exception(f"Distiller {distil_path} not implemented in visualization yet.")
    return distilled_filters, distilled_actions, distilled_extras, agent_pos #distilled_extras is None if not applicable, filters already reformed to gameboard size


def main():
    print("Episode runner")

    experiment, timestep_path, distil_path = handle_selection()

    episode_str = timestep_path / "0"
    replay = experiment.replay_episode(episode_str)
    episode = replay.episode
    
    action_names, extras_meaning = get_env_infos(experiment)

    if distil_path is not None: 
        distilled_filters, distilled_actions, distilled_extras, agent_pos = handle_distillation(episode, distil_path)
        viewer = HeatmapXFrameViewer(replay.frames, episode.n_agents, agent_pos, distilled_actions, action_names, distilled_filters, distilled_extras, extras_meaning)
    else:
        viewer = FrameViewer(replay.frames)
    viewer.show()

if __name__ == "__main__":
    main()