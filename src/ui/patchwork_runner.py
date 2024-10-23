import cv2
import base64
import numpy as np
from marl.models import Experiment, ReplayEpisode

def decode_b64_image(base64_str: str) -> np.ndarray:
    if base64_str is None:
        return ""
    base64_str = base64_str.split(',')[-1]
    image_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return image


exp_path_base = "D:\\logs\\"
exp_name = "XMARL_MO_L5_AGENTID"
exp_path = f"{exp_path_base}{exp_name}"
exp = Experiment.load(exp_path)
run_str = "run_2024-10-13_20-15-55.817905_seed=0"
timestamp = "1000000"
episode_str = f"{exp_path}\\{run_str}\\test\\{timestamp}\\0"
episode = exp.replay_episode(episode_str)
episode.episode.rewards



for frame in episode.frames:
    img_frame = decode_b64_image(frame)
    cv2.imshow("frame",img_frame)
    cv2.waitKey(0)
cv2.destroyAllWindows()