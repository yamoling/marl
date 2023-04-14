from typing import Type
import inspect
from marl.models import RLAlgo
from marl import qlearning, policy_gradient



def _is_algo_predicate(cls: Type[RLAlgo]) -> bool:
    return inspect.isclass(cls) and issubclass(cls, RLAlgo) and not inspect.isabstract(cls)

def _get_all_algos() -> dict[str, Type[RLAlgo]]:
    qlearning_algos = inspect.getmembers(qlearning, _is_algo_predicate)
    policy_gradient_algos = inspect.getmembers(policy_gradient, _is_algo_predicate)
    classes = {name: cls for name, cls in qlearning_algos }
    classes.update({name: cls for name, cls in policy_gradient_algos })
    return classes

def from_summary(summary: dict[str, ]) -> RLAlgo:
    clss = ALGO_REGISTRY[summary["name"]]
    return clss.from_summary(summary)

def register(model: Type[RLAlgo]):
    """Register a RLAlgo"""
    ALGO_REGISTRY[model.__name__] = model

ALGO_REGISTRY: dict[str, Type[RLAlgo]] = _get_all_algos()
