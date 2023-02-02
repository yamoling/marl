from typing import TypeVar
import re

T = TypeVar("T")

import torch

def get_device(device: str="auto") -> torch.device:
    """Get the given device"""
    if device == "auto" or device == "" or device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return get_device(device)
    return torch.device(device)


def defaults_to(value: T | None, default: T)  -> T:
    """
    Shortcut to retrieve a default value.
    
    NB: by using this method, the default object is created in every case! Do not use this method
    if the instanciation is expensive.
    """
    if value is not None:
        return value
    return default
    

def alpha_num_order(string):
   """ Returns all numbers on 5 digits to let sort the string with numeric order.
   Ex: alphaNumOrder("a6b12.125")  ==> "a00006b00012.00125"
   """
   return ''.join([format(int(x), '08d') if x.isdigit()
                   else x for x in re.split(r'(\d+)', string)])