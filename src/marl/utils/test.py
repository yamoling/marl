from typing import Type
import inspect
from marl.models import RLAlgo
from marl import qlearning, policy_gradient

ALGO_REGISTRY: dict[str, Type[RLAlgo]] = {}

def _is_algo_predicate(cls: Type[RLAlgo]) -> bool:
    return inspect.isclass(cls) and issubclass(cls, RLAlgo) and not inspect.isabstract(cls)

def _get_all_algos() -> dict[str, Type[RLAlgo]]:
    classes = {}
    for name, cls in inspect.getmembers(qlearning, _is_algo_predicate):
        classes[name] = cls
    for name, cls in inspect.getmembers(policy_gradient, _is_algo_predicate):
        classes[name] = cls
    from marl.utils.random_algo import RandomAgent
    classes[RandomAgent.__name__] = RandomAgent
    return classes

def from_summary(summary: dict[str, ]) -> RLAlgo:
    clss = ALGO_REGISTRY[summary["name"]]
    return clss.from_summary(summary)

def register(algo: Type[RLAlgo]):
    """Register a RLAlgo"""
    ALGO_REGISTRY[algo.__name__] = algo

ALGO_REGISTRY.update(_get_all_algos())


