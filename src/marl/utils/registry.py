import inspect
import pprint
from types import ModuleType
from typing import Any, Callable, Type, TypeVar

from .serializable import Serializable

T = TypeVar("T", bound=Serializable)


class NoSuchClass(Exception):
    pass


class Registry(dict[str, Type[T]]):
    def register(self, cls: Type[T]):
        if cls.__name__ in self and self[cls.__name__] != cls:
            raise Exception(f"Class {cls.__name__} is already registered with an other match: {self[cls.__name__]}.")
        self[cls.__name__] = cls

    def from_dict(self, data: dict[str,]) -> T:
        try:
            cls = self[data.pop("__type__")]
        except KeyError:
            raise Exception(
                f"\n{pprint.pformat(data)}\nThere is no field '__type__' in the above json. Cannot retrieve the name of the class to load."
            )
        try:
            return cls.from_dict(data)
        except KeyError:
            raise NoSuchClass(
                f"""Could not find any class with name '{data['__type__']}' in {T.__class__.__name__} registry.
Use register({cls.__name__}) to register your class."""
            )
        except TypeError as e:
            raise TypeError(str(e) + "\nDid you forget to save the appropriate fields in the summary?")


RegisterFn = Callable[[Type[T]], None]
FromSummaryFn = Callable[[dict[str, Any]], T]


def make_registry(super_type: Type[T], modules_to_inspect: list[ModuleType]):
    def is_target_type(cls: Type[T]) -> bool:
        return inspect.isclass(cls) and issubclass(cls, super_type) and not inspect.isabstract(cls)

    registry = Registry[T]()
    for module in modules_to_inspect:
        for _name, cls in inspect.getmembers(module, is_target_type):
            registry.register(cls)
    return registry.register, registry.from_dict
