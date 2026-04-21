import base64
import os
import random
import re
from typing import Callable, Optional, TypeVar

import cv2
import numpy as np
import torch
from marlenv import MARLEnv, Observation


def seed(seed: int, env: Optional[MARLEnv] = None):
    """Seeds `random`, `numpy`, `torch` and the environment (if provided) with the given seed value."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if env is not None:
        env.seed(seed)


T = TypeVar("T")


def defaults_to(value: T | None, default: Callable[[], T]) -> T:
    """
    Shortcut to retrieve a default value if the given one is None.
    """
    if value is not None:
        return value
    return default()


def alpha_num_order(string):
    """Returns all numbers on 5 digits to let sort the string with numeric order.
    Ex: alphaNumOrder("a6b12.125")  ==> "a00006b00012.00125"
    """
    return "".join([format(int(x), "08d") if x.isdigit() else x for x in re.split(r"(\d+)", string)])


def encode_b64_image(image: np.ndarray) -> str:
    return base64.b64encode(cv2.imencode(".jpg", image)[1]).decode("ascii")


def hash_ndarray(data: np.ndarray) -> int:
    return hash(data.tobytes())


def obs_to_hashes(obs: Observation):
    obs_data = np.concatenate((obs.data, obs.extras), axis=-1)
    return [hash_ndarray(o) for o in obs_data]
