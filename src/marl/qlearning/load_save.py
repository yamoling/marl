from typing import Type
import inspect
from .qlearning import IQLearning


def _is_qlearning_predicate(cls: Type[IQLearning]) -> bool:
    return inspect.isclass(cls) and issubclass(cls, IQLearning) and cls != IQLearning

def _get_all_algos() -> dict[str, Type[IQLearning]]:
    from marl import qlearning
    classes = inspect.getmembers(qlearning, _is_qlearning_predicate)
    return {name: cls for name, cls in classes }

def from_summary(summary: dict[str, ]) -> IQLearning:
    return ALL_ALGOS[summary["name"]].from_summary(summary)

def register(model: Type[IQLearning]):
    """Register a RLAlgo"""
    ALL_ALGOS[model.__name__] = model

ALL_ALGOS: dict[str, Type[IQLearning]] = _get_all_algos()
