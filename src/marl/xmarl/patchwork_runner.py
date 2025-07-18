from marl.models import Experiment
from marlenv.models import Episode

import pathlib
import os
import numpy as np

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
    extra = False
    if "distil" in exp_cont:
        distil_list = [distil for distil in os.listdir(exp_path/"distil") if "distil" in distil]
        if len(distil_list) > 0:
            print("Distillation(s) of the experiment's model found, do you want to visualize it? \n[Y/N]")
            yn = input()
            if str.upper(yn) == "Y": distil_path = get_selection(exp_path/"distil", distil_list)
            if "individual" in str(distil_path):
                for d in os.listdir(distil_path):
                    if "extra" in d: extra = True
                if extra:
                    print("For individual agent distillation there is a version with and without extras. Do you want extras (Y) or not (n)?")
                    yn = input()
                    if str.upper(yn) != "Y": extra = False

    return experiment, timestep_path, distil_path, extra

def handle_distillation(episode: Episode, distil_path: pathlib.Path, extra: bool):
    if "sdt" in str(distil_path):
        print("SDT has two types of explanations to provide, choose by inputting the index: \n0: Forward (Traverse greediest path to action) - 1: Backward (Filters of path to original action)")
        e_type = input() == "1"
        if "individual" in str(distil_path):
            distilled_filters = []
            distilled_extras = []
            distilled_actions = []
            agent_pos = []
            for ag in range(episode.n_agents):
                distiller = SoftDecisionTree.load(distil_path/f"ag{ag}_sdt_distil{"_extra" if extra else ""}.pkl")
                df, de, da, ap = distiller.distil_episode(episode, e_type)
                distilled_filters.append(df)
                distilled_extras.append(de)
                distilled_actions.append(da)
                agent_pos.append(ap)
            distilled_filters= np.transpose(np.array(distilled_filters), (1,0,2,3,4))
            if len(distilled_extras) != 0: distilled_extras = np.transpose(np.array(distilled_extras).squeeze(), (1,0,2,3))
            if e_type: distilled_actions = np.transpose(np.array(distilled_actions), (1,0,2))
            else: distilled_actions = np.transpose(np.array(distilled_actions), (1,0,2,3))
            agent_pos = np.transpose(np.array(agent_pos), (1,0,2))
        else: 
            distiller = SoftDecisionTree.load(distil_path)
            distilled_filters, distilled_extras, distilled_actions, agent_pos = distiller.distil_episode(episode, e_type) # Give shape for obs and extras to be computed there, or reshape after, need to separate extras that's annoying
    else: raise Exception(f"Distiller {distil_path} not implemented in visualization yet.")
    return distilled_filters, distilled_actions, distilled_extras, agent_pos #distilled_extras is None if not applicable, filters already reformed to gameboard size


def main():
    print("Episode runner")

    experiment, timestep_path, distil_path, extra = handle_selection()

    episode_str = timestep_path / "0"
    replay = experiment.replay_episode(episode_str)
    episode = replay.episode
    
    action_names, extras_meaning = get_env_infos(experiment)

    if distil_path is not None: 
        distilled_filters, distilled_actions, distilled_extras, agent_pos = handle_distillation(episode, distil_path, extra)
        # Insert 7x7 obs in original frame
        if experiment.env.observation_shape[1:] == (7,7): # TODO: Hardcoded, no way to get original world size without assuming env specific type
            agent_pos = np.array(episode.other["ag_pos"])
            n_obs = np.zeros(distilled_filters.shape[0:-2]+(12,13)) # hard coded world size for LLE L6)
            for t in range(episode.episode_len):
                for a in range(episode.n_agents):
                    filt = distilled_filters[t, a] # shape (11, 7, 7)
                    x, y = agent_pos[t, a]
                    
                    # Compute insertion indices
                    x_start = x - 3
                    y_start = y - 3
                    x_end = x_start + 7
                    y_end = y_start + 7

                    # Clip off overlap between 7x7 and bounds gameboard
                    x_s, x_e = max(x_start, 0), min(x_end, 12)
                    y_s, y_e = max(y_start, 0), min(y_end, 13)

                    # Corresponding slice in filter
                    fx_s, fx_e = x_s - x_start, x_e - x_start
                    fy_s, fy_e = y_s - y_start, y_e - y_start

                    # Insert into the output
                    n_obs[t, a, :, x_s:x_e, y_s:y_e] = filt[:, fx_s:fx_e, fy_s:fy_e]
            distilled_filters = n_obs

        viewer = HeatmapXFrameViewer(replay.frames, episode.n_agents, agent_pos, distilled_actions, action_names, distilled_filters, distilled_extras, extras_meaning)
    else:
        viewer = FrameViewer(replay.frames)
    viewer.show()

if __name__ == "__main__":
    main()