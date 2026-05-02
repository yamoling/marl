from abc import abstractmethod
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Self, Type, Union, get_args, get_origin

import numpy as np
import orjson
import torch
from lle import World


def default_serialization(obj):
    """Default behaviour for orjson serialization"""
    match obj:
        case torch.nn.Parameter() as param:
            return f"Parameter(shape={param.shape}, dtype={param.dtype})"
        case set():
            return list(obj)
        case torch.optim.Optimizer() as optim:
            return {
                "name": optim.__class__.__name__,
                "param_groups": [{k: v for k, v in group.items()} for group in optim.param_groups],
            }
        case np.ndarray():
            return obj.tolist()
        case np.signedinteger():
            return int(obj)
        case np.floating():
            return float(obj)
        case World():
            return obj.world_string
    raise TypeError(f"Type {type(obj)} is not serializable")


def get_subclass_map(base_class: Type):
    """
    Recursively finds all subclasses and maps their names to the class object.
    """
    mapping = {base_class.__name__: base_class}
    for subclass in base_class.__subclasses__():
        mapping[subclass.__name__] = subclass
        # Recurse in case there are subclasses of subclasses
        mapping.update(get_subclass_map(subclass))
    return mapping


def get_subclass_with_name(base_class: Type, class_name: str) -> Type | None:
    """
    Retrieve the subclass whose name is `class_name`, if if exist.

    **Note:** the class provided as argument is not considered to be a subclass of itself.
    """
    for subclass in base_class.__subclasses__():
        if subclass.__name__ == class_name:
            return subclass
        # Recurse in case there are subclasses of subclasses
        result = get_subclass_with_name(subclass, class_name)
        if result is not None:
            return result
    return None


@dataclass
class Serializable[T]:
    name: str = field(init=False)

    def __post_init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def make(self) -> T: ...

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """
        Recursively build the configuration from a dictionary.
        Child objects that are serializable are deserialized thanks to their `from_dict` method.
        """
        # If the "name" field is no longer there, it means that a parent class has already handled this case.
        class_name = d.pop("name", cls.__name__)
        if class_name != cls.__name__:
            subtype = get_subclass_with_name(cls, class_name)
            if subtype is None:
                raise KeyError(f"Unknown subclass {class_name} for {cls.__name__}")
            return subtype.from_dict(d)

        # Iterate on all fields to identify complex ones that require deserialization
        for f in fields(cls):
            # Don't bother for simple types, None values or values not in `d`
            if f.type in (int, str, float, bool) or d.get(f.name) is None:
                continue
            actual_type = f.type
            # For union types like `X | None`, get the `X` type
            if get_origin(actual_type) is Union:
                # Filter out NoneType to find the actual class
                union_types = [a for a in get_args(actual_type) if a is not type(None)]
                if len(union_types) > 1:
                    raise NotImplementedError(f"Union type {actual_type} has more than one non-None type")
                actual_type = union_types[0]
            # If the resulting type is a Serializable subclass, then deserialize it
            if isinstance(actual_type, type) and issubclass(actual_type, Serializable):
                d[f.name] = actual_type.from_dict(d[f.name])
        return cls(**d)

    @classmethod
    def from_json(cls, data: bytes) -> Self:
        """Build the configuration from a JSON file."""
        d = orjson.loads(data)
        return cls.from_dict(d)

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return orjson.dumps(self.to_dict())
