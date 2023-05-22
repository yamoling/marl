"""Intrinsic reward module."""

from .ir_module import IRModule
from .random_network_distillation import RandomNetworkDistillation

from marl.utils.registry import make_registry

register, from_summary = make_registry(IRModule, [random_network_distillation])


