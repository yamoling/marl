import base64
import cv2
import numpy as np
from typing import Callable, TypeVar, Literal
import re
import socket
from contextlib import closing
   
def get_available_port(port=8000) -> int:
    MIN_PORT = 1024
    MAX_PORT = 65535
    port = min(max(port, MIN_PORT), MAX_PORT)
    for port in range(port, MAX_PORT):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            if sock.connect_ex(("0.0.0.0", port)) != 0:
                # Port is available
                return port
    raise RuntimeError("No available port found")

T = TypeVar("T")

import torch

def get_device(device: Literal["auto", "cuda", "cpu"]="auto") -> torch.device:
    """Get the given device"""
    if device == "auto" or device == "" or device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return get_device(device)
    return torch.device(device)


def defaults_to(value: T | None, default: Callable[[], T])  -> T:
    """
    Shortcut to retrieve a default value if the given one is None.
    """
    if value is not None:
        return value
    return default()
    

def alpha_num_order(string):
   """ Returns all numbers on 5 digits to let sort the string with numeric order.
   Ex: alphaNumOrder("a6b12.125")  ==> "a00006b00012.00125"
   """
   return ''.join([format(int(x), '08d') if x.isdigit()
                   else x for x in re.split(r'(\d+)', string)])


def encode_b64_image(image: np.ndarray) -> str:
    if image is None:
        return ""
    return base64.b64encode(cv2.imencode(".jpg", image)[1]).decode("ascii")
