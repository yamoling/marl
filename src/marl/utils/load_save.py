from typing import Type
import inspect
import marl


def _is_algo_predicate(cls: Type[marl.RLAlgo]) -> bool:
    return inspect.isclass(cls) and issubclass(cls, marl.RLAlgo) and cls != marl.RLAlgo

def _get_all_algos() -> dict[str, Type[marl.RLAlgo]]:
    classes = inspect.getmembers(marl, _is_algo_predicate)
    return {name: cls for name, cls in classes }

def from_summary(summary: dict[str, ]) -> marl.RLAlgo:
    return ALL_ALGOS[summary["name"]].from_summary(summary)

def register(model: Type[marl.RLAlgo]):
    """Register a RLAlgo"""
    ALL_ALGOS[model.__name__] = model

ALL_ALGOS: dict[str, Type[marl.RLAlgo]] = _get_all_algos()
