from typing import Type, TypeVar, Callable
from types import ModuleType
import inspect
from .summarizable import Summarizable


T = TypeVar("T", bound=Summarizable)


class NoSuchClass(Exception):
    pass


class Registry(dict[str, Type[T]]):
    def register(self, cls: Type[T]):
        self[cls.__name__] = cls

    def from_summary(self, summary: dict[str, ]) -> T:
        try:
            clss = self[summary.pop("name")]
            return clss.from_summary(summary)
        except KeyError:
            raise NoSuchClass(f"Could not find any class with name '{summary['name']}' in {T.__class__.__name__} registry.\nUse register({clss.__name__}) to register your class.")
        except TypeError as e:
            raise TypeError(str(e) + "\nDid you forget to save the appropriate fields in the summary?")


RegisterFunction = Callable[[Type[T]], None]
FromSummaryFunction = Callable[[dict[str, ]], T]

def make_registry(super_type: Type[T], modules_to_inspect: list[ModuleType]):
    def is_target_type(cls: Type[T]) -> bool:
        return inspect.isclass(cls) and issubclass(cls, super_type) and not inspect.isabstract(cls)
    registry = Registry[T]()
    for module in modules_to_inspect:
        for _name, clss in inspect.getmembers(module, is_target_type):
            registry.register(clss)
    return registry.register, registry.from_summary

