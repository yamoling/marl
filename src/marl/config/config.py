from dataclasses import asdict, dataclass
from typing import ClassVar, Type, get_args

from marl.utils import Serializable


@dataclass
class Config[T](Serializable):
    _concrete_class: ClassVar[Type | None] = None

    def __init_subclass__(cls) -> None:
        if cls._concrete_class is None:
            cls.__name__
        return super().__init_subclass__()

    def make(self) -> T:
        if not hasattr(self, "__orig_class__"):
            raise TypeError(f"{self.__class__.__name__} must be instantiated with a generic type !")
        generic_types = get_args(self.__orig_class__)  # type: ignore (__orig_class__ is not detected)
        if len(generic_types) != 1:
            raise TypeError(f"{self.__class__.__name__} must be instantiated with exactly one generic type, but got {generic_types}")
        target_class: Type[T] = generic_types[0]
        return target_class(**asdict(self))
