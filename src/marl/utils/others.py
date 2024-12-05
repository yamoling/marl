import base64
import cv2
import numpy as np
from typing import Callable, Optional, TypeVar
import re
from marlenv import MARLEnv
from marlenv.models.env import ActionSpaceType as A, RewardType as R, StateType as S, ObsType as O


def seed(seed_value: int, env: Optional[MARLEnv[A, O, S, R]] = None):
    import torch
    import random
    import numpy as np

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if env is not None:
        env.seed(seed_value)


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
    if image is None:
        return ""
    return base64.b64encode(cv2.imencode(".jpg", image)[1]).decode("ascii")  # type: ignore
