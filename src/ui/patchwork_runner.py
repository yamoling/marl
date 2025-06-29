import cv2
import base64
import numpy as np
from marl.models import Experiment, ReplayEpisode
import typed_argparse as tap

import pathlib
import os


LOG_PATH = pathlib.Path("logs")

def decode_b64_image(base64_str: str) -> np.ndarray:
    if base64_str is None:
        return ""
    base64_str = base64_str.split(',')[-1]
    image_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return image


def get_selection(cur_dir, file_list):
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

def main():
    print("Episode runner")
    exp_list = os.listdir(LOG_PATH)

    exp_path = get_selection(LOG_PATH, exp_list)     
    experiment = Experiment.load(exp_path)
    if experiment.logdir != exp_path:
            experiment.logdir = exp_path
    print()

    run_list = os.listdir(exp_path)
    run_list = [run for run in run_list if "run" in run]
    run_path = get_selection(exp_path, run_list)
    print()

    test_path = run_path / "test"
    test_list = os.listdir(test_path)
    timestep_path = get_selection(test_path, test_list)
    print()

    episode_str = timestep_path / "0"
    episode = experiment.replay_episode(episode_str)
    episode.episode.rewards

    for frame in episode.frames:
        img_frame = decode_b64_image(frame)
        cv2.imshow("frame",img_frame)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()