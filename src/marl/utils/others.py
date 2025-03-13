import base64
import cv2
import numpy as np
from typing import Callable, Optional, TypeVar
import re
from marlenv import MARLEnv, ActionSpace
import torch


def seed[A, AS: ActionSpace](seed_value: int, env: Optional[MARLEnv[A, AS]] = None):
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


def default_serialization(obj):
    """Default behaviour for orjson serialization"""
    match obj:
        case torch.nn.Parameter() as param:
            return f"Parameter(shape={param.shape}, dtype={param.dtype})"
        case set():
            return list(obj)
        case torch.optim.Optimizer() as optim:
            return {
                "name": optim.__class__.__name__,
                "param_groups": [{k: v for k, v in group.items()} for group in optim.param_groups],
            }
        case np.signedinteger():
            return int(obj)
        case np.floating():
            return float(obj)
    return str(obj)
