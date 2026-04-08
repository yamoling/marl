from typing import Type

import numpy as np
import torch
from lle import World


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
        case np.ndarray():
            return obj.tolist()
        case np.signedinteger():
            return int(obj)
        case np.floating():
            return float(obj)
        case World():
            return obj.world_string
    raise TypeError(f"Type {type(obj)} is not serializable")


def get_subclass_map(base_class: Type) -> dict[str, Type]:
    """
    Recursively finds all subclasses and maps their names to the class object.
    """
    mapping = {}
    for subclass in base_class.__subclasses__():
        mapping[subclass.__name__] = subclass
        # Recurse in case there are subclasses of subclasses
        mapping.update(get_subclass_map(subclass))
    return mapping
