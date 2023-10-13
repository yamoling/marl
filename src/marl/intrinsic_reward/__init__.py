"""Intrinsic reward module."""


from . import random_network_distillation
from .ir_module import IRModule
from .random_network_distillation import RandomNetworkDistillation


__all__ = ["IRModule", "RandomNetworkDistillation", "register", "load"]
