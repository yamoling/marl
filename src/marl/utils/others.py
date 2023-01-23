from typing import TypeVar

T = TypeVar("T")

import torch

def get_device(device: str="auto") -> torch.device:
    """Get the given device"""
    if device == "auto" or device == "" or device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return get_device(device)
    return torch.device(device)


def defaults_to(value: T | None, default: T)  -> T:
    """Shortcut to retrieve a default value"""
    if value is not None:
        return value
    return default
    