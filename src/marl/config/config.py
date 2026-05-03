from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

from marl.utils import Serializable


@dataclass
class Config[T](Serializable):
    name: str = field(init=False)

    def __post_init__(self):
        name = self.__class__.__name__.removesuffix("Config").removesuffix("Conf")
        if len(name) == 0:
            name = self.__class__.__name__
        self.name = name

    @abstractmethod
    def make(self) -> T: ...

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        # Remove the "name" field because it is not part of the constructor arguments.
        # We must add a default value such that it does not fail if `from_dict` is called recursively
        # due to child class dispatching.
        d.pop("name", None)
        return super().from_dict(d)
