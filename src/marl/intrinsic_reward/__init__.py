"""Intrinsic reward module."""

from marl.utils.registry import make_registry

from . import random_network_distillation
from .ir_module import IRModule
from .random_network_distillation import RandomNetworkDistillation

register, load = make_registry(IRModule, [random_network_distillation])

__all__ = ["IRModule", "RandomNetworkDistillation", "register", "load"]
