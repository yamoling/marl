from abc import abstractmethod
from typing import Any

from marl.utils import Serializable

DISPLAY_NAME = "dislay-name"


class Config[T](Serializable):
    @abstractmethod
    def make(self) -> T: ...

    def to_dict(self):
        d = super().to_dict()
        d[DISPLAY_NAME] = self.__class__.__name__.removesuffix("Config").removesuffix("Conf")
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        # Remove the DISPLAY_NAME field because it is not part of the constructor arguments.
        # We add a default value such that it does not fail if `from_dict` is called recursively due to child class dispatching.
        d.pop(DISPLAY_NAME, None)
        return super().from_dict(d)
